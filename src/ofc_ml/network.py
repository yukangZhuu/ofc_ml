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
                # nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                # nn.Dropout(dropout)
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
        # freqs: (n_freq, 1) -> (n_freq, 1, 1)
        # u: (B, out_features) -> (1, B, out_features)
        fu = self.freqs.unsqueeze(2) * u.unsqueeze(0)
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


class SpectralMixingLayer(nn.Module):
    """
    轻量级频域混合层 (类似 FNO 但更简单)
    对输入的"特征维度"进行频域全局混合，捕获全局依赖关系
    """
    def __init__(self, hidden_dim: int, n_spectral_modes: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        # FFT 后的频域维度
        self.freq_dim = hidden_dim // 2 + 1
        self.n_modes = min(n_spectral_modes, self.freq_dim)  # 只保留低频模态
        
        # 频域可学习权重（复数）- 对每个频率模态
        # 形状: (n_modes,) - 每个频率一个复数权重
        self.weights_real = nn.Parameter(torch.randn(self.n_modes) * 0.02)
        self.weights_imag = nn.Parameter(torch.randn(self.n_modes) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, hidden_dim)
        对 hidden_dim 这个维度做频域混合
        """
        batch_size = x.shape[0]
        
        # 1. FFT: 转到频域 (对每个样本的 hidden_dim 维度做 FFT)
        x_ft = torch.fft.rfft(x, dim=1)  # (batch, freq_dim) 复数
        
        # 2. 频域混合：只对低频模态应用可学习权重
        out_ft = x_ft.clone()
        
        # 构造复数权重 (n_modes,)
        weights_complex = torch.complex(self.weights_real, self.weights_imag)
        
        # 频域加权: (batch, n_modes) * (n_modes,) 广播
        out_ft[:, :self.n_modes] = x_ft[:, :self.n_modes] * weights_complex.unsqueeze(0)
        
        # 高频部分衰减
        if self.freq_dim > self.n_modes:
            out_ft[:, self.n_modes:] = out_ft[:, self.n_modes:] * 0.1  # 高频衰减
        
        # 3. IFFT: 转回空间域
        out = torch.fft.irfft(out_ft, n=self.hidden_dim, dim=1)  # (batch, hidden_dim)
        
        return out


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


class HybridFNOKANPredictor(nn.Module):
    """
    混合 FNO + FourierKAN 架构
    在网络中间插入频域混合层，捕获全局通道间依赖
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=None,
        dropout: float = 0.2,
        use_residual: bool = True,
        n_frequencies: int = 4,
        n_spectral_modes: int = 16,
        use_spectral_mixing: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 128, 64]

        self.use_residual = use_residual
        self.use_spectral_mixing = use_spectral_mixing
        
        # 前期 FourierKAN blocks（特征提取）
        self.blocks_early = nn.ModuleList()
        self.proj_early = nn.ModuleList()
        
        # 后期 FourierKAN blocks（精细调整）
        self.blocks_late = nn.ModuleList()
        self.proj_late = nn.ModuleList()
        
        # 决定在哪一层插入频域混合
        split_idx = len(hidden_dims) // 2  # 在中间插入
        
        # 构建前期层
        prev = input_dim
        for i, h in enumerate(hidden_dims[:split_idx]):
            self.blocks_early.append(FourierKANBlock(prev, h, dropout=dropout, n_frequencies=n_frequencies))
            if use_residual and prev != h:
                p = nn.Linear(prev, h, bias=False)
                nn.init.xavier_uniform_(p.weight)
                self.proj_early.append(p)
            else:
                self.proj_early.append(nn.Identity())
            prev = h
        
        # 频域混合层
        if self.use_spectral_mixing:
            self.spectral_mixing = SpectralMixingLayer(prev, n_spectral_modes=n_spectral_modes)
            print(f"[HybridFNOKAN] Inserted SpectralMixingLayer at layer {split_idx}, dim={prev}, modes={n_spectral_modes}")
        
        # 构建后期层
        for i, h in enumerate(hidden_dims[split_idx:]):
            self.blocks_late.append(FourierKANBlock(prev, h, dropout=dropout, n_frequencies=n_frequencies))
            if use_residual and prev != h:
                p = nn.Linear(prev, h, bias=False)
                nn.init.xavier_uniform_(p.weight)
                self.proj_late.append(p)
            else:
                self.proj_late.append(nn.Identity())
            prev = h

        # 输出层
        self.head = nn.Linear(prev, output_dim)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x, mask=None):
        out = x
        
        # 前期处理
        for block, proj in zip(self.blocks_early, self.proj_early):
            if self.use_residual:
                out = block(out) + proj(out)
            else:
                out = block(out)
        
        # 频域混合（全局通道交互）
        if self.use_spectral_mixing:
            out = out + self.spectral_mixing(out)  # 残差连接
        
        # 后期处理
        for block, proj in zip(self.blocks_late, self.proj_late):
            if self.use_residual:
                out = block(out) + proj(out)
            else:
                out = block(out)

        # 输出
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