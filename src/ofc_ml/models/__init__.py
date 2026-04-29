"""Model registry for EDFA digital twin experiments.

All architectures share the same (B, input_dim) -> (B, output_dim=95) signature
and accept an optional `mask` of shape (B, 95) that is multiplied against the
output (training loss masks are separate).

Usage
-----
    from ofc_ml.models import build_model
    model = build_model(cfg.model, input_dim, output_dim)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ..configs.schema import ModelConfig
from .hybrid_fno_kan import HybridFNOKANPredictor
from .mlp import MLPPredictor
from .cnn1d import CNN1DPredictor
from .transformer import TransformerPredictor
from .wang_dnn import WangDNNPredictor
from .physics_baseline import PhysicsBaselinePredictor


def build_model(cfg: ModelConfig, input_dim: int, output_dim: int = 95) -> nn.Module:
    name = cfg.name.lower().strip()
    if name == "hybrid_fno_kan":
        return HybridFNOKANPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_residual=cfg.use_residual,
            n_frequencies=cfg.n_frequencies,
            spectral_freq_ratio=cfg.spectral_freq_ratio,
            use_spectral_mixing=cfg.use_spectral_mixing,
            use_fourier_kan=cfg.use_fourier_kan,
        )
    if name == "mlp":
        return MLPPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_residual=cfg.use_residual,
        )
    if name == "cnn1d":
        return CNN1DPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            channels=cfg.cnn_channels,
            kernel_size=cfg.cnn_kernel_size,
            dropout=cfg.dropout,
        )
    if name == "transformer":
        return TransformerPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=cfg.tr_d_model,
            nhead=cfg.tr_nhead,
            num_layers=cfg.tr_num_layers,
            dim_feedforward=cfg.tr_dim_feedforward,
            dropout=cfg.dropout,
        )
    if name == "wang_dnn":
        return WangDNNPredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )
    if name == "physics_baseline":
        # Non-trainable analytical reference. `input_dim` is unused by the
        # module itself but kept in the signature for symmetry.
        return PhysicsBaselinePredictor(output_dim=output_dim)
    raise ValueError(f"Unknown model name: {cfg.name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "HybridFNOKANPredictor",
    "MLPPredictor",
    "CNN1DPredictor",
    "TransformerPredictor",
    "WangDNNPredictor",
    "PhysicsBaselinePredictor",
    "build_model",
    "count_parameters",
]
