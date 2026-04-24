"""Offline evaluation against `test_labels.csv`.

Metric conventions
------------------
All metrics are computed on the activated (mask == 1) channel entries only.
The mask comes from `test_features.csv` (`DUT_WSS_activated_channel_index_*`),
matching training-time semantics; never from `test_labels.csv` whose zeros are
merely placeholders for inactive channels.

Kaggle Score (competition metric, reproduced offline):
    per-row  a_i = mean_j |y - y_hat|  over activated j
             s_i = std_j  |y - y_hat|  over activated j
    dataset  MAE = mean_i a_i
             Std = mean_i s_i
             errors = { |y - y_hat| for all activated (i, j) }
             T95 = max(0, quantile(errors, 0.95) - MAE - 0.5)
             Tmax= max(0, max(errors)             - quantile(errors, 0.95) - 0.7)
    Score = MAE + 0.3 * T95 + 0.1 * Tmax + 0.15 * Std

The thresholds 0.5 and 0.7 match the `tau` and `beta` defaults in
`KaggleScoreLoss` (see `model.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


N_CHANNELS = 95
KAGGLE_TAU = 0.5   # T95 threshold
KAGGLE_BETA = 0.7  # Tmax threshold


# ---------------------------------------------------------------------- #
# Core metric functions                                                  #
# ---------------------------------------------------------------------- #
def _compute_core_metrics(errors: np.ndarray, per_row_mae: np.ndarray, per_row_std: np.ndarray) -> Dict[str, float]:
    """Given already-masked flat error array and per-row stats, compute the metric set.

    Units
    -----
    Every metric inherits its unit from the label `calculated_gain_spectra_*`,
    which is **dB** (gain = EDFA_output - EDFA_input, both in dBm).  To make
    the unit explicit downstream we publish a `units` sub-dict and duplicate
    every scalar under an explicit `_dB` (or `_dB2` for MSE) suffix.  The
    original keys are retained for backwards compatibility.
    """
    if errors.size == 0:
        nan = float("nan")
        empty = {
            "n_samples": 0,
            "n_activated": 0,
            "MAE": nan, "RMSE": nan, "MSE": nan, "Std": nan,
            "T95": nan, "Tmax": nan, "KaggleScore": nan,
            "MAE_dB": nan, "RMSE_dB": nan, "MSE_dB2": nan, "Std_dB": nan,
            "T95_dB": nan, "Tmax_dB": nan, "KaggleScore_dB": nan,
            "units": {"MAE": "dB", "RMSE": "dB", "MSE": "dB^2", "Std": "dB",
                      "T95": "dB", "Tmax": "dB", "KaggleScore": "dB"},
        }
        return empty

    mae = float(per_row_mae.mean())
    std = float(per_row_std.mean())
    mse = float((errors ** 2).mean())
    rmse = float(np.sqrt(mse))
    p95 = float(np.quantile(errors, 0.95))
    emax = float(errors.max())
    t95 = max(0.0, p95 - mae - KAGGLE_TAU)
    tmax = max(0.0, emax - p95 - KAGGLE_BETA)
    kaggle = mae + 0.3 * t95 + 0.1 * tmax + 0.15 * std
    return {
        "n_samples": int(per_row_mae.size),
        "n_activated": int(errors.size),
        # Original keys (kept for backward compatibility)
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "Std": std,
        "T95": t95,
        "Tmax": tmax,
        "KaggleScore": kaggle,
        # Unit-tagged duplicates
        "MAE_dB": mae,
        "RMSE_dB": rmse,
        "MSE_dB2": mse,
        "Std_dB": std,
        "T95_dB": t95,
        "Tmax_dB": tmax,
        "KaggleScore_dB": kaggle,
        "units": {
            "MAE": "dB", "RMSE": "dB", "MSE": "dB^2", "Std": "dB",
            "T95": "dB", "Tmax": "dB", "KaggleScore": "dB",
        },
    }


def _compute_subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """y_true, y_pred, mask are (N, 95) arrays; mask is 0/1."""
    mask_bool = mask > 0
    abs_err = np.abs(y_pred - y_true)
    abs_err_masked = abs_err * mask  # zero out non-activated

    # Per-row stats (over activated channels only)
    # Guard against rows with zero activated channels.
    k_per_row = mask.sum(axis=1)
    per_row_safe_k = np.where(k_per_row > 0, k_per_row, 1)
    per_row_mae = abs_err_masked.sum(axis=1) / per_row_safe_k
    # std (population) of activated errors per row:
    mean_broadcast = per_row_mae[:, None]
    sq = ((abs_err_masked - mean_broadcast) ** 2) * mask
    per_row_var = sq.sum(axis=1) / per_row_safe_k
    per_row_std = np.sqrt(np.clip(per_row_var, 0.0, None))

    # Only keep rows that have any activated channel to compute averages.
    valid_rows = k_per_row > 0
    per_row_mae = per_row_mae[valid_rows]
    per_row_std = per_row_std[valid_rows]

    errors = abs_err[mask_bool]  # flat, only activated entries
    return _compute_core_metrics(errors, per_row_mae, per_row_std)


# ---------------------------------------------------------------------- #
# Top-level entry point                                                  #
# ---------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    overall: Dict[str, float]
    by_usage: Dict[str, Dict[str, float]]
    by_category: Dict[str, Dict[str, float]]
    by_edfa_type: Dict[str, Dict[str, float]]

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "overall": self.overall,
            "by_usage": self.by_usage,
            "by_category": self.by_category,
            "by_edfa_type": self.by_edfa_type,
        }


def _mask_from_features(test_features: pd.DataFrame) -> np.ndarray:
    mask_cols = sorted(
        [c for c in test_features.columns if "DUT_WSS_activated_channel_index" in c],
        key=lambda x: int(x.split("_")[-1]),
    )
    if len(mask_cols) != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS} mask columns in test_features, got {len(mask_cols)}"
        )
    return test_features[mask_cols].values.astype(np.float32)


def _target_cols(df: pd.DataFrame) -> List[str]:
    cols = sorted(
        [c for c in df.columns if "calculated_gain_spectra_" in c],
        key=lambda x: int(x.split("_")[-1]),
    )
    if len(cols) != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} gain-spectra columns, got {len(cols)}")
    return cols


def compute_metrics(
    predictions: pd.DataFrame | np.ndarray,
    test_labels: pd.DataFrame,
    test_features: pd.DataFrame,
    per_usage: bool = True,
    per_category: bool = True,
    per_edfa_type: bool = True,
) -> EvaluationResult:
    """Compute all evaluation metrics.

    Parameters
    ----------
    predictions
        Either a DataFrame with `ID + calculated_gain_spectra_*` (submission
        format) or a (N, 95) numpy array aligned with `test_features`.
    test_labels
        Ground truth DataFrame read from `test_labels.csv`; must contain `ID`,
        `Usage`, and 95 `calculated_gain_spectra_*` columns.
    test_features
        Corresponding `test_features.csv` DataFrame (used for mask + strata).
    """
    # Align predictions to test_features by ID.
    lbl_cols = _target_cols(test_labels)

    if isinstance(predictions, pd.DataFrame):
        if "ID" not in predictions.columns:
            raise ValueError("Prediction DataFrame must have an 'ID' column.")
        merged = test_features[["ID"]].merge(predictions[["ID"] + lbl_cols], on="ID", how="left")
        y_pred = merged[lbl_cols].values.astype(np.float32)
    else:
        y_pred = np.asarray(predictions, dtype=np.float32)
        if y_pred.shape != (len(test_features), N_CHANNELS):
            raise ValueError(
                f"Array prediction shape {y_pred.shape} != expected {(len(test_features), N_CHANNELS)}"
            )

    merged_lbl = test_features[["ID"]].merge(test_labels[["ID", "Usage"] + lbl_cols], on="ID", how="left")
    y_true = merged_lbl[lbl_cols].values.astype(np.float32)
    mask = _mask_from_features(test_features)

    overall = _compute_subset_metrics(y_true, y_pred, mask)

    by_usage: Dict[str, Dict[str, float]] = {}
    if per_usage and "Usage" in merged_lbl.columns:
        for usage, idx in merged_lbl.groupby("Usage").groups.items():
            sel = np.array(idx.tolist())
            by_usage[str(usage)] = _compute_subset_metrics(y_true[sel], y_pred[sel], mask[sel])

    by_category: Dict[str, Dict[str, float]] = {}
    if per_category and "Category" in test_features.columns:
        for cat, idx in test_features.groupby("Category").groups.items():
            sel = np.array(idx.tolist())
            by_category[str(cat)] = _compute_subset_metrics(y_true[sel], y_pred[sel], mask[sel])

    by_edfa_type: Dict[str, Dict[str, float]] = {}
    if per_edfa_type and "EDFA_type" in test_features.columns:
        for et, idx in test_features.groupby("EDFA_type").groups.items():
            sel = np.array(idx.tolist())
            by_edfa_type[str(et)] = _compute_subset_metrics(y_true[sel], y_pred[sel], mask[sel])

    return EvaluationResult(
        overall=overall,
        by_usage=by_usage,
        by_category=by_category,
        by_edfa_type=by_edfa_type,
    )


def evaluate_submission_file(
    submission_path: str | Path,
    test_labels_path: str | Path,
    test_features_path: str | Path,
    **kwargs,
) -> EvaluationResult:
    sub = pd.read_csv(submission_path)
    labels = pd.read_csv(test_labels_path)
    features = pd.read_csv(test_features_path)
    return compute_metrics(sub, labels, features, **kwargs)


__all__ = [
    "EvaluationResult",
    "compute_metrics",
    "evaluate_submission_file",
]
