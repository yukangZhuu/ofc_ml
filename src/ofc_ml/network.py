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


class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        
        # Init weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class ResNetPredictor(nn.Module):
    """
    ResNet-MLP Architecture:
    - Robust deep MLP with residual connections
    - LayerNorm for stability
    """
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        hidden_dims=[256, 256, 256, 256], 
        dropout=0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        
        # ResNet Blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            dim = hidden_dims[i]
            # If next layer has different dim, we need a projection (not implemented here for simplicity, 
            # assuming hidden_dims are all same or handled by block if we wanted)
            # For this simple ResNet, we assume constant width or handle transition carefully.
            # Here we just use ResNetBlock which keeps dim same.
            # If user provides different dims, we'll project between them.
            
            self.blocks.append(ResNetBlock(dim, dropout))
            
            # Transition layer if next dim is different (and not last layer)
            if i < len(hidden_dims) - 1 and hidden_dims[i] != hidden_dims[i+1]:
                trans = nn.Linear(hidden_dims[i], hidden_dims[i+1])
                nn.init.xavier_uniform_(trans.weight)
                self.blocks.append(trans)

        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], output_dim)
        )
        for m in self.output_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
            
        out = self.output_head(x)
        
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
    使用固定比例保留低频模态，确保不同维度下物理意义一致
    """
    def __init__(self, hidden_dim: int, freq_ratio: float = 0.5):
        """
        Args:
            hidden_dim: 特征维度
            freq_ratio: 保留的频率比例 (0-1之间)，例如0.5表示保留低50%的频率
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freq_ratio = freq_ratio
        
        # FFT 后的频域维度
        self.freq_dim = hidden_dim // 2 + 1
        
        # 根据比例计算要保留的模态数（至少保留1个）
        self.n_modes = max(1, int(self.freq_dim * freq_ratio))
        
        # 频域可学习权重（复数）- 对每个频率模态
        # 形状: (n_modes,) - 每个频率一个复数权重
        self.weights_real = nn.Parameter(torch.randn(self.n_modes) * 0.02)
        self.weights_imag = nn.Parameter(torch.randn(self.n_modes) * 0.02)
        
        # 打印调试信息
        print(f"[SpectralMixing] dim={hidden_dim}, freq_dim={self.freq_dim}, "
              f"n_modes={self.n_modes} ({self.n_modes/self.freq_dim*100:.1f}% of frequencies)")
        
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
        
        # # 高频部分衰减
        # if self.freq_dim > self.n_modes:
        #     out_ft[:, self.n_modes:] = out_ft[:, self.n_modes:] * 0.1  # 高频衰减
        
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
    在每两个FourierKAN层之间插入频域混合层，捕获全局通道间依赖
    使用固定频率保留比例，确保各层物理意义一致
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=None,
        dropout: float = 0.2,
        use_residual: bool = True,
        n_frequencies: int = 4,
        spectral_freq_ratio: float = 0.5,
        use_spectral_mixing: bool = True,
    ):
        """
        Args:
            spectral_freq_ratio: 频域混合层保留的频率比例 (0-1)，默认0.5表示保留低50%频率
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 128, 128]

        self.use_residual = use_residual
        self.use_spectral_mixing = use_spectral_mixing
        
        # FourierKAN blocks
        self.blocks = nn.ModuleList()
        self.proj = nn.ModuleList()
        
        # 频域混合层（在每两个FourierKAN之间插入）
        self.spectral_mixings = nn.ModuleList()
        
        # 构建交替的 FourierKAN Block 和 SpectralMixingLayer
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            # 添加 FourierKAN Block
            self.blocks.append(FourierKANBlock(prev, h, dropout=dropout, n_frequencies=n_frequencies))
            
            # 添加投影层（用于残差连接）
            if use_residual and prev != h:
                p = nn.Linear(prev, h, bias=False)
                nn.init.xavier_uniform_(p.weight)
                self.proj.append(p)
            else:
                self.proj.append(nn.Identity())
            
            # 在每个FourierKAN Block后添加频域混合层（除了最后一层）
            if self.use_spectral_mixing and i < len(hidden_dims) - 1:
                spectral_layer = SpectralMixingLayer(h, freq_ratio=spectral_freq_ratio)
                self.spectral_mixings.append(spectral_layer)
                print(f"[HybridFNOKAN] Inserted SpectralMixingLayer after block {i}")
            
            prev = h

        # 输出层
        self.head = nn.Linear(prev, output_dim)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)
        
        print(f"[HybridFNOKAN] Total architecture: {len(self.blocks)} FourierKAN Blocks + {len(self.spectral_mixings)} SpectralMixing Layers")

    def forward(self, x, mask=None):
        out = x
        
        # 交替处理：FourierKAN Block → SpectralMixing → FourierKAN Block → ...
        spectral_idx = 0
        for i, (block, proj) in enumerate(zip(self.blocks, self.proj)):
            # FourierKAN Block with residual
            if self.use_residual:
                out = block(out) + proj(out)
            else:
                out = block(out)
            
            # 如果不是最后一层，且启用了频域混合，则应用SpectralMixing
            if self.use_spectral_mixing and i < len(self.blocks) - 1:
                out = out + self.spectral_mixings[spectral_idx](out)  # 残差连接
                spectral_idx += 1

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