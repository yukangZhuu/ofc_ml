"""Non-trainable physics-baseline predictor.

Returns a `(B, output_dim)` zero tensor for any input, so when the runner's
`predict_test()` is called with `predict_absolute=False` the inference path
collapses to:

    out = baseline + zero_pred  -> baseline
    out = out * mask

i.e. exactly the analytical gain/tilt baseline `target_gain + target_gain_tilt
* (47 - j) / 94`, masked to activated channels.

The class registers no learnable parameters; pairing it with `stages: []` in
the experiment YAML means `orchestrate()` runs no training and the runner
still produces the canonical `metrics.json` / `submission.csv` artefacts.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PhysicsBaselinePredictor(nn.Module):
    """Identity-zero predictor used as the analytical-baseline reference."""

    def __init__(self, output_dim: int = 95):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = x.new_zeros((x.shape[0], self.output_dim))
        if mask is not None:
            out = out * mask
        return out
