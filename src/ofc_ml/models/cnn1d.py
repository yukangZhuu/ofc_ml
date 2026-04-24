"""1D CNN baseline (M-3): channel-as-sequence.

Data layout assumption
----------------------
The preprocessor in `features.py` emits columns in the order:
    [ num (4), cat (one-hot, variable), spectra (95), mask (95) ]
so the last 190 dims always encode the per-channel signal (spectra, mask)
and the leading `input_dim - 190` dims are scalar features (numerics + one-hot).

We reshape the last 190 dims into a (B, 2, 95) tensor (spectra, mask) and tile
the scalar features into additional constant channels.  The resulting 1D
representation is fed through a small Conv1d stack that preserves the sequence
length `L=95`, and a final 1x1 Conv1d projects each channel position to the
scalar offset prediction.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


N_CHANNELS = 95


class CNN1DPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = N_CHANNELS,
        channels: Optional[List[int]] = None,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 64, 128, 128]
        if output_dim != N_CHANNELS:
            raise ValueError(f"CNN1D baseline expects output_dim={N_CHANNELS}, got {output_dim}")

        self.input_dim = int(input_dim)
        self.n_scalar = input_dim - 2 * N_CHANNELS
        if self.n_scalar < 0:
            raise ValueError(
                f"input_dim={input_dim} smaller than 2*{N_CHANNELS}; CNN1D baseline requires "
                f"the preprocessor to include both spectra and mask columns."
            )

        # in_channels = spectra(1) + mask(1) + broadcast(scalar)(n_scalar)
        in_channels = 2 + self.n_scalar
        layers: List[nn.Module] = []
        prev_c = in_channels
        for c in channels:
            layers.append(nn.Conv1d(prev_c, c, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_c = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(prev_c, 1, kernel_size=1)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def _split_and_reshape(self, x: torch.Tensor) -> torch.Tensor:
        """(B, input_dim) -> (B, 2 + n_scalar, 95)."""
        if self.n_scalar > 0:
            scalars = x[:, : self.n_scalar]                 # (B, n_scalar)
            per_channel = x[:, self.n_scalar:]              # (B, 2*95)
        else:
            scalars = x.new_zeros((x.shape[0], 0))
            per_channel = x
        spectra = per_channel[:, :N_CHANNELS].unsqueeze(1)  # (B, 1, 95)
        mask = per_channel[:, N_CHANNELS:].unsqueeze(1)     # (B, 1, 95)
        if self.n_scalar > 0:
            scalars = scalars.unsqueeze(-1).expand(-1, -1, N_CHANNELS)  # (B, n_scalar, 95)
            return torch.cat([spectra, mask, scalars], dim=1)
        return torch.cat([spectra, mask], dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self._split_and_reshape(x)        # (B, C, 95)
        z = self.body(z)                      # (B, C', 95)
        out = self.head(z).squeeze(1)         # (B, 95)
        if mask is not None:
            out = out * mask
        return out
