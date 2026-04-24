"""HybridFNOKAN predictor: FourierKAN blocks + SpectralMixing.

This module is the canonical home of the architecture.  `src/ofc_ml/network.py`
re-exports the public names for backward compatibility with pre-refactor code.

Switches (new)
--------------
- `use_spectral_mixing`: if False, drop all SpectralMixing layers (ablation A-A1).
- `use_fourier_kan`:     if False, replace FourierKAN blocks with plain
  Linear+LayerNorm+GELU+Dropout blocks (ablation A-A2).  Residual connections
  and the SpectralMixing layers are preserved.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class FourierKANLayer(nn.Module):
    """FourierKAN layer (lightweight).

    u = Wx + b
    output = u + Σ_f [ sin(f*u) * A_sin[f] + cos(f*u) * A_cos[f] ]
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

        freqs = torch.arange(1, self.n_frequencies + 1, dtype=torch.float32).view(self.n_frequencies, 1)
        self.register_buffer("freqs", freqs, persistent=False)

        self.amp_sin = nn.Parameter(torch.zeros(self.n_frequencies, out_features))
        self.amp_cos = nn.Parameter(torch.zeros(self.n_frequencies, out_features))
        nn.init.normal_(self.amp_sin, mean=0.0, std=0.02)
        nn.init.normal_(self.amp_cos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.linear(x)
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


class MLPBlock(nn.Module):
    """Linear + LayerNorm + GELU + Dropout (drop-in replacement for FourierKANBlock).

    Used by ablation A-A2 (w/o FourierKAN), and by the MLPPredictor baseline.
    Parameter count is approximately equal to a FourierKANBlock of the same
    in/out dims (FourierKANBlock adds only 2 * n_freq * out_features params
    from amp_sin/amp_cos, which is small relative to the Linear weight).
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.0)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class SpectralMixingLayer(nn.Module):
    """Lightweight frequency-domain mixing (FNO-like).

    Applies rFFT over the feature (hidden) dimension, multiplies the lowest
    `freq_ratio` modes by learnable complex weights, and inverts back.
    """

    def __init__(self, hidden_dim: int, freq_ratio: float = 0.5):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.freq_ratio = float(freq_ratio)
        self.freq_dim = hidden_dim // 2 + 1
        self.n_modes = max(1, int(self.freq_dim * freq_ratio))

        self.weights_real = nn.Parameter(torch.randn(self.n_modes) * 0.02)
        self.weights_imag = nn.Parameter(torch.randn(self.n_modes) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft(x, dim=1)
        out_ft = x_ft.clone()
        weights = torch.complex(self.weights_real, self.weights_imag)
        out_ft[:, : self.n_modes] = x_ft[:, : self.n_modes] * weights.unsqueeze(0)
        return torch.fft.irfft(out_ft, n=self.hidden_dim, dim=1)


class HybridFNOKANPredictor(nn.Module):
    """FourierKAN stack interleaved with optional SpectralMixing layers.

    Architecture (per block i):
        y = block_i(x) + proj_i(x)                      # block residual
        y = y + spectral_mixing_i(y)                    # spectral residual (optional)

    After the last block, a `Linear(prev, output_dim)` head predicts the target.
    When `mask` is provided at `forward`, the output is element-wise multiplied
    by it (used to zero out non-activated channels).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_residual: bool = True,
        n_frequencies: int = 4,
        spectral_freq_ratio: float = 0.5,
        use_spectral_mixing: bool = True,
        use_fourier_kan: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 128, 128]

        self.use_residual = use_residual
        self.use_spectral_mixing = use_spectral_mixing
        self.use_fourier_kan = use_fourier_kan

        self.blocks = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.spectral_mixings = nn.ModuleList()

        prev = input_dim
        for i, h in enumerate(hidden_dims):
            if use_fourier_kan:
                self.blocks.append(FourierKANBlock(prev, h, dropout=dropout, n_frequencies=n_frequencies))
            else:
                self.blocks.append(MLPBlock(prev, h, dropout=dropout))

            if use_residual and prev != h:
                p = nn.Linear(prev, h, bias=False)
                nn.init.xavier_uniform_(p.weight)
                self.proj.append(p)
            else:
                self.proj.append(nn.Identity())

            if self.use_spectral_mixing and i < len(hidden_dims) - 1:
                self.spectral_mixings.append(SpectralMixingLayer(h, freq_ratio=spectral_freq_ratio))

            prev = h

        self.head = nn.Linear(prev, output_dim)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = x
        spectral_idx = 0
        for i, (block, proj) in enumerate(zip(self.blocks, self.proj)):
            if self.use_residual:
                out = block(out) + proj(out)
            else:
                out = block(out)

            if self.use_spectral_mixing and i < len(self.blocks) - 1:
                out = out + self.spectral_mixings[spectral_idx](out)
                spectral_idx += 1

        out = self.head(out)
        if mask is not None:
            out = out * mask
        return out
