"""Same-size MLP baseline (M-2).

Stack of MLP blocks with the same depth/width as the HybridFNOKAN main model,
but WITHOUT FourierKAN, WITHOUT SpectralMixing.  Residual connections are
preserved for a fair comparison against the main architecture -- the sole
variable is the per-block non-linearity and the frequency mixing.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .hybrid_fno_kan import MLPBlock


class MLPPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_residual: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 128, 128]
        self.use_residual = use_residual

        self.blocks = nn.ModuleList()
        self.proj = nn.ModuleList()

        prev = input_dim
        for h in hidden_dims:
            self.blocks.append(MLPBlock(prev, h, dropout=dropout))
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

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
