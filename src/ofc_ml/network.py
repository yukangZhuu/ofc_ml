import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

def compute_baseline_gain(target_gain, target_gain_tilt, num_channels=95):
    """
    Compute baseline gain for each channel using target_gain and target_gain_tilt.
    
    Args:
        target_gain: scalar or tensor, baseline gain value at center channel
        target_gain_tilt: scalar or tensor, tilt value across channels
        num_channels: int, number of channels (default 95)
    
    Returns:
        baseline: numpy array or tensor of shape (batch_size, num_channels), baseline gain for each channel
    """
    center_idx = (num_channels - 1) / 2  # Center index (47 for 95 channels)
    
    # Create channel indices
    channel_indices = np.arange(num_channels)
    
    # Handle both scalar and batch inputs
    if np.isscalar(target_gain):
        baseline = target_gain + target_gain_tilt * (center_idx - channel_indices) / (num_channels - 1)
    else:
        # Batch processing
        batch_size = len(target_gain)
        baseline = np.zeros((batch_size, num_channels))
        for i in range(batch_size):
            baseline[i] = target_gain[i] + target_gain_tilt[i] * (center_idx - channel_indices) / (num_channels - 1)
    
    return baseline

class OFCDataset(Dataset):
    def __init__(self, X, y, target_gain, target_gain_tilt, mask=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.target_gain = torch.FloatTensor(target_gain)
        self.target_gain_tilt = torch.FloatTensor(target_gain_tilt)
        if mask is not None:
            self.mask = torch.FloatTensor(mask)
        else:
            self.mask = None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.mask is not None:
            return self.X[idx], self.y[idx], self.target_gain[idx], self.target_gain_tilt[idx], self.mask[idx]
        else:
            return self.X[idx], self.y[idx], self.target_gain[idx], self.target_gain_tilt[idx], torch.ones(self.y.shape[1])

class SimpleGainPredictor(nn.Module):
    def __init__(self, input_dim=99, output_dim=95, hidden_dims=[256, 128], dropout=0.3):
        super(SimpleGainPredictor, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, mask=None):
        out = self.network(x)
        if mask is not None:
            out = out * mask
        return out


class FourierKANLayer(nn.Module):
    """
    FourierKAN layer (lightweight):
    - Project input to out_features with a linear layer -> u
    - Apply multiple fixed Fourier frequencies on u: sin(f*u), cos(f*u)
    - Learn per-frequency, per-dimension amplitudes and sum them back to out_features.
    """

    def __init__(self, in_features: int, out_features: int, n_frequencies: int = 4, bias: bool = True):
        super().__init__()
        if n_frequencies < 1:
            raise ValueError("n_frequencies must be >= 1")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_frequencies = int(n_frequencies)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)

        # Fixed frequencies: 1,2,...,n_frequencies
        freqs = torch.arange(1, self.n_frequencies + 1, dtype=torch.float32).view(self.n_frequencies, 1)
        self.register_buffer("freqs", freqs, persistent=False)

        # Learnable amplitudes for each frequency and each output dimension
        self.amp_sin = nn.Parameter(torch.zeros(self.n_frequencies, out_features))
        self.amp_cos = nn.Parameter(torch.zeros(self.n_frequencies, out_features))
        nn.init.normal_(self.amp_sin, mean=0.0, std=0.02)
        nn.init.normal_(self.amp_cos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.linear(x)  # (B, out_features)
        # (n_freq, B, out) via broadcasting
        fu = self.freqs * u.unsqueeze(0)
        sin_terms = torch.sin(fu) * self.amp_sin.unsqueeze(1)
        cos_terms = torch.cos(fu) * self.amp_cos.unsqueeze(1)
        return u + (sin_terms + cos_terms).sum(dim=0)


class FourierKANBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, n_frequencies: int = 4):
        super().__init__()
        self.kan = FourierKANLayer(in_features, out_features, n_frequencies=n_frequencies)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kan(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class FourierKANGainPredictor(nn.Module):
    """
    Deeper FourierKAN predictor for offset regression.
    Input: (B, input_dim) -> Output: (B, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=None,
        dropout: float = 0.2,
        use_residual: bool = True,
        n_frequencies: int = 4,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 128, 64]

        self.use_residual = use_residual

        self.blocks = nn.ModuleList()
        self.proj = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.blocks.append(FourierKANBlock(prev, h, dropout=dropout, n_frequencies=n_frequencies))
            if use_residual and prev != h:
                p = nn.Linear(prev, h, bias=False)
                nn.init.xavier_uniform_(p.weight)
                self.proj.append(p)
            else:
                self.proj.append(nn.Identity())
            prev = h

        self.head = nn.Linear(prev, output_dim)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x, mask=None):
        out = x
        for block, proj in zip(self.blocks, self.proj):
            if self.use_residual:
                out = block(out) + proj(out)
            else:
                out = block(out)

        out = self.head(out)
        if mask is not None:
            out = out * mask
        return out

# Backward-compatible aliases (old SimpleKAN naming)
SimpleKANLayer = FourierKANLayer
SimpleKANBlock = FourierKANBlock
SimpleKANGainPredictor = FourierKANGainPredictor

class TargetNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, y):
        self.mean = y.mean(axis=0)
        self.std = y.std(axis=0)
        self.std[self.std == 0] = 1.0
    
    def transform(self, y):
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been fitted yet")
        return (y - self.mean) / self.std
    
    def inverse_transform(self, y):
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been fitted yet")
        return y * self.std + self.mean
    
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)