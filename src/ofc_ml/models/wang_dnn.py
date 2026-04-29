"""Wang-style DNN baseline (global-adapted).

Implements the dense feed-forward EDFA gain-spectrum DNN used by
Wang et al. 2023 (Sec. 5, Fig. 9), adapted to the global multi-EDFA
input feature schema used everywhere else in this project.

Architecture
------------
Body: a stack of `[Linear -> BatchNorm1d -> ELU (-> Dropout)]` blocks with
hidden widths `[256, 128, 128, 128]` by default (Wang's published widths).
Head: `Linear(prev, output_dim=95)` with no activation, masked by the
activated-channel indicator in `forward` when provided.

All `Linear` layers use Kaiming-normal initialisation with `nonlinearity='relu'`,
which is the standard Kaiming setting in PyTorch and the closest match to ELU
that `torch.nn.init` exposes; biases are zero-initialised.

Faithful and adapted choices, plus where this baseline differs from the Wang
TL paper, are documented in `docs/wang_dnn_vs_ours_spec.md`.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class WangDNNPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 95,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 128, 128]
        if not hidden_dims:
            raise ValueError("WangDNNPredictor requires at least one hidden layer")

        layers: List[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            linear = nn.Linear(prev, h)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h
        self.body = nn.Sequential(*layers)

        self.head = nn.Linear(prev, output_dim)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="relu")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.body(x)
        out = self.head(z)
        if mask is not None:
            out = out * mask
        return out
