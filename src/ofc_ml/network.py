"""Backwards-compatibility shim.

The canonical implementation of the HybridFNOKAN architecture now lives in
`ofc_ml.models.hybrid_fno_kan`.  This module re-exports the public names so
existing imports (e.g. `from ofc_ml.network import HybridFNOKANPredictor`)
keep working.  It also retains the small helpers that are shared by training
code: `OFCDataset`, `compute_baseline_gain`, `TargetNormalizer`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .models.hybrid_fno_kan import (
    FourierKANBlock,
    FourierKANLayer,
    HybridFNOKANPredictor,
    MLPBlock,
    SpectralMixingLayer,
)

__all__ = [
    "FourierKANLayer",
    "FourierKANBlock",
    "MLPBlock",
    "SpectralMixingLayer",
    "HybridFNOKANPredictor",
    "OFCDataset",
    "compute_baseline_gain",
    "TargetNormalizer",
]


def compute_baseline_gain(target_gain, target_gain_tilt, num_channels: int = 95):
    """Per-channel baseline gain from (target_gain, target_gain_tilt)."""
    center_idx = (num_channels - 1) / 2
    channel_indices = np.arange(num_channels)

    if np.isscalar(target_gain):
        return target_gain + target_gain_tilt * (center_idx - channel_indices) / (num_channels - 1)

    target_gain = np.asarray(target_gain).reshape(-1, 1)
    target_gain_tilt = np.asarray(target_gain_tilt).reshape(-1, 1)
    channel_indices = channel_indices.reshape(1, -1)
    tilt_factor = (center_idx - channel_indices) / (num_channels - 1)
    return target_gain + target_gain_tilt * tilt_factor


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
        return (
            self.X[idx],
            self.y[idx],
            self.target_gain[idx],
            self.target_gain_tilt[idx],
            torch.ones(self.y.shape[1]),
        )


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
