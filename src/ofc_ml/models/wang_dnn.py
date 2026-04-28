"""Wang et al. 2023 reference DNN baseline (M-0).

Reproduces the EDFA gain-spectrum DNN described in
"Open EDFA gain spectrum dataset and its applications in data-driven EDFA gain
modeling", Wang et al. 2023, Section 5.A and Fig. 9.

Reference architecture (per the paper):
    193 -> 256 -> 128 -> 128 -> 128 -> 95
    each hidden block: Linear -> BatchNorm1d -> ELU
    output: Linear (no activation)
    init  : Kaiming normal

Adaptation in this codebase
---------------------------
The original paper trains *one DNN per EDFA device* on a 193-dim feature vector
(`g0`, `P_in`, `P_out`, 95 input-spectrum samples, 95 channel-loading
indicators).  In our setup we train a **single global model** for all EDFAs and
reuse the project-wide preprocessor; that adds the `target_gain_tilt` scalar
plus a small one-hot encoding of `EDFA_type` and `edfa_index` (~204 dims total
in our preprocessor's output).  The only structural change is therefore the
width of the first linear layer: `Linear(input_dim, 256)`.  Everything else
(hidden widths, BN+ELU blocks, Kaiming initialisation, masked-MSE loss, the
three-phase transfer protocol) is preserved exactly.

We make this trade because: (a) the Kaggle test set contains an `unseen`
device split, on which a per-device model has no training data; (b) keeping
the same input pipeline isolates the architectural / training-protocol
differences from feature-engineering differences when comparing M-0 against
the other models.

The classifier `head` is exposed as `self.head` so the Wang-style three-phase
transfer protocol can freeze and re-initialise just the output layer.
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
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 128, 128]

        layers: List[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ELU())
            prev = h
        self.body = nn.Sequential(*layers)
        # The output layer is intentionally exposed as `self.head` so the
        # Wang-style three-phase transfer can freeze / re-init just this part.
        self.head = nn.Linear(prev, int(output_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def reset_head(self) -> None:
        """Re-initialise the output layer with Kaiming normal (Wang Phase A)."""
        nn.init.kaiming_normal_(self.head.weight)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.body(x)
        out = self.head(z)
        if mask is not None:
            out = out * mask
        return out
