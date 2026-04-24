"""Transformer baseline (M-4): channel-as-token.

Each of the 95 channels becomes a token carrying:
    [spectra_value, mask, scalar_features_broadcast]
tokens are projected to `d_model`, a learnable positional embedding is added,
and a small Transformer encoder mixes them.  A linear head per token emits the
scalar offset prediction.

Data layout assumption matches CNN1DPredictor: flat input vector is
    [ scalars (input_dim - 190), spectra (95), mask (95) ]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


N_CHANNELS = 95


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = N_CHANNELS,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if output_dim != N_CHANNELS:
            raise ValueError(f"Transformer baseline expects output_dim={N_CHANNELS}, got {output_dim}")

        self.input_dim = int(input_dim)
        self.n_scalar = input_dim - 2 * N_CHANNELS
        if self.n_scalar < 0:
            raise ValueError(
                f"input_dim={input_dim} smaller than 2*{N_CHANNELS}; Transformer baseline requires "
                f"the preprocessor to include both spectra and mask columns."
            )

        token_in = 2 + self.n_scalar
        self.token_proj = nn.Linear(token_in, d_model)
        nn.init.xavier_uniform_(self.token_proj.weight)
        if self.token_proj.bias is not None:
            nn.init.constant_(self.token_proj.bias, 0.0)

        self.pos_emb = nn.Parameter(torch.zeros(1, N_CHANNELS, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_scalar > 0:
            scalars = x[:, : self.n_scalar]
            per_channel = x[:, self.n_scalar:]
        else:
            scalars = x.new_zeros((x.shape[0], 0))
            per_channel = x
        spectra = per_channel[:, :N_CHANNELS].unsqueeze(-1)  # (B, 95, 1)
        mask_vals = per_channel[:, N_CHANNELS:].unsqueeze(-1)  # (B, 95, 1)
        if self.n_scalar > 0:
            scalars = scalars.unsqueeze(1).expand(-1, N_CHANNELS, -1)  # (B, 95, n_scalar)
            tok = torch.cat([spectra, mask_vals, scalars], dim=-1)
        else:
            tok = torch.cat([spectra, mask_vals], dim=-1)
        return tok  # (B, 95, token_in)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tok = self._tokenize(x)
        z = self.token_proj(tok) + self.pos_emb  # (B, 95, d_model)
        # Use `src_key_padding_mask` to skip attention for inactive channels
        # when mask is available.  Shape expected by PyTorch: (B, L), True=pad.
        key_padding = None
        if mask is not None:
            key_padding = mask <= 0  # non-activated channels are "padding"
        z = self.encoder(z, src_key_padding_mask=key_padding)
        out = self.head(z).squeeze(-1)  # (B, 95)
        if mask is not None:
            out = out * mask
        return out
