"""Data loading helpers.

The new config-driven path is `load_datasets(cfg)`, which returns the three
splits the training pipeline needs: COSMOS (pretraining), Kaggle train
(fine-tuning), and Kaggle test (final evaluation).  `cfg.data.cosmos_ratio`
and `cfg.data.kaggle_ratio` support the data-scale ablation.

The legacy functions `load_data` and `load_data_separate` are kept as thin
wrappers over the legacy module-level constants in `config.py` for
backwards-compatibility with the pre-refactor entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import config as cfg
from .configs.schema import ExperimentConfig


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def subsample(
    df_X: pd.DataFrame,
    df_y: pd.DataFrame,
    ratio: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Random subsample X/y together, preserving row correspondence."""
    if ratio is None or ratio >= 1.0:
        return df_X, df_y
    if ratio <= 0.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    n = len(df_X)
    k = max(1, int(round(n * ratio)))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:k]
    idx.sort()
    return df_X.iloc[idx].reset_index(drop=True), df_y.iloc[idx].reset_index(drop=True)


def _load_csv_pair(features_path: Path, labels_path: Path, label: str):
    print(f"[data] Loading {label} features from {features_path.name}...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    print(f"[data]   {label}: features {X.shape}, labels {y.shape}")
    return X, y


# ---------------------------------------------------------------------- #
# COSMOS sharded / single-file loader                                    #
# ---------------------------------------------------------------------- #
def _load_cosmos_dataset_paths(features_path: Path, labels_path: Path, root: Path):
    if features_path.exists() and labels_path.exists():
        return _load_csv_pair(features_path, labels_path, "COSMOS")

    root = Path(root)
    feat_files = sorted(root.glob("**/train_features.csv"))
    label_files = sorted(root.glob("**/train_labels.csv"))
    if not feat_files or not label_files:
        raise FileNotFoundError(
            f"Could not find cosmos train csvs under {root}. "
            "Expected train_features.csv/train_labels.csv either at root or in subfolders."
        )

    labels_by_parent = {p.parent: p for p in label_files}
    pairs = []
    for f in feat_files:
        l = labels_by_parent.get(f.parent)
        if l is None:
            raise FileNotFoundError(f"Missing train_labels.csv for {f.parent}")
        pairs.append((f, l))

    print(f"[data] Loading COSMOS train from {root} ({len(pairs)} shards)...")
    Xs, ys = [], []
    base_X_cols = None
    base_y_cols = None
    for f, l in pairs:
        Xi = pd.read_csv(f)
        yi = pd.read_csv(l)
        if base_X_cols is None:
            base_X_cols = list(Xi.columns)
            base_y_cols = list(yi.columns)
        else:
            if list(Xi.columns) != base_X_cols:
                raise ValueError(f"train_features schema mismatch in {f}")
            if list(yi.columns) != base_y_cols:
                raise ValueError(f"train_labels schema mismatch in {l}")
        Xs.append(Xi)
        ys.append(yi)

    return pd.concat(Xs, axis=0, ignore_index=True), pd.concat(ys, axis=0, ignore_index=True)


# ---------------------------------------------------------------------- #
# New config-driven loader                                               #
# ---------------------------------------------------------------------- #
@dataclass
class LoadedDatasets:
    cosmos_features: pd.DataFrame
    cosmos_labels: pd.DataFrame
    kaggle_features: pd.DataFrame
    kaggle_labels: pd.DataFrame
    test_features: pd.DataFrame
    # Present only when test_labels_path exists (ground truth, post-competition).
    test_labels: Optional[pd.DataFrame] = None


def load_datasets(cfg_exp: ExperimentConfig) -> LoadedDatasets:
    """Load COSMOS, Kaggle train, Kaggle test (optionally test_labels) per cfg.

    Subsampling is applied to COSMOS / Kaggle train according to
    `cfg.data.cosmos_ratio` / `cfg.data.kaggle_ratio`, using `cfg.seed`.
    Test features & test labels are never subsampled.
    """
    d = cfg_exp.data

    cosmos_X, cosmos_y = _load_cosmos_dataset_paths(
        d.cosmos_train_features_path, d.cosmos_train_labels_path, d.cosmos_data_dir
    )
    kaggle_X, kaggle_y = _load_csv_pair(d.train_features_path, d.train_labels_path, "Kaggle")
    test_X = pd.read_csv(d.test_features_path)
    print(f"[data] Test features: {test_X.shape}")

    test_y = None
    if d.test_labels_path and Path(d.test_labels_path).exists():
        test_y = pd.read_csv(d.test_labels_path)
        print(f"[data] Test labels:   {test_y.shape}")

    # Schema check
    if list(kaggle_X.columns) != list(cosmos_X.columns):
        raise ValueError("Features schema mismatch between Kaggle and COSMOS datasets.")
    if list(kaggle_y.columns) != list(cosmos_y.columns):
        raise ValueError("Labels schema mismatch between Kaggle and COSMOS datasets.")

    # Subsample
    if d.cosmos_ratio < 1.0:
        cosmos_X, cosmos_y = subsample(cosmos_X, cosmos_y, d.cosmos_ratio, seed=cfg_exp.seed)
        print(f"[data] COSMOS subsampled to ratio={d.cosmos_ratio}: {cosmos_X.shape}")
    if d.kaggle_ratio < 1.0:
        kaggle_X, kaggle_y = subsample(kaggle_X, kaggle_y, d.kaggle_ratio, seed=cfg_exp.seed)
        print(f"[data] Kaggle subsampled to ratio={d.kaggle_ratio}: {kaggle_X.shape}")

    return LoadedDatasets(
        cosmos_features=cosmos_X,
        cosmos_labels=cosmos_y,
        kaggle_features=kaggle_X,
        kaggle_labels=kaggle_y,
        test_features=test_X,
        test_labels=test_y,
    )


# ---------------------------------------------------------------------- #
# Legacy API (kept for backwards compatibility)                          #
# ---------------------------------------------------------------------- #
def _load_cosmos_dataset():
    return _load_cosmos_dataset_paths(
        Path(cfg.COSMOS_TRAIN_FEATURES_PATH),
        Path(cfg.COSMOS_TRAIN_LABELS_PATH),
        Path(cfg.COSMOS_DATA_DIR),
    )


def load_data():
    """Legacy: returns (train_features, train_labels, test_features) per cfg.DATASET_USE."""
    dataset_use = str(cfg.DATASET_USE).lower().strip()
    if dataset_use not in {"kaggle", "cosmos", "both"}:
        raise ValueError(f"Unsupported DATASET_USE={cfg.DATASET_USE!r}.")

    if dataset_use == "kaggle":
        train_X = pd.read_csv(cfg.TRAIN_FEATURES_PATH)
        train_y = pd.read_csv(cfg.TRAIN_LABELS_PATH)
    elif dataset_use == "cosmos":
        train_X, train_y = _load_cosmos_dataset()
    else:
        k_X = pd.read_csv(cfg.TRAIN_FEATURES_PATH)
        k_y = pd.read_csv(cfg.TRAIN_LABELS_PATH)
        c_X, c_y = _load_cosmos_dataset()
        if list(k_X.columns) != list(c_X.columns):
            raise ValueError("train_features schema mismatch between kaggle and cosmos datasets.")
        if list(k_y.columns) != list(c_y.columns):
            raise ValueError("train_labels schema mismatch between kaggle and cosmos datasets.")
        train_X = pd.concat([k_X, c_X], axis=0, ignore_index=True)
        train_y = pd.concat([k_y, c_y], axis=0, ignore_index=True)
    test_X = pd.read_csv(cfg.TEST_FEATURES_PATH)
    return train_X, train_y, test_X


def load_data_separate():
    """Legacy: (cosmos_features, cosmos_labels, kaggle_features, kaggle_labels, test_features)."""
    cosmos_X, cosmos_y = _load_cosmos_dataset()
    kaggle_X = pd.read_csv(cfg.TRAIN_FEATURES_PATH)
    kaggle_y = pd.read_csv(cfg.TRAIN_LABELS_PATH)
    if list(kaggle_X.columns) != list(cosmos_X.columns):
        raise ValueError("Features schema mismatch between Kaggle and COSMOS datasets.")
    if list(kaggle_y.columns) != list(cosmos_y.columns):
        raise ValueError("Labels schema mismatch between Kaggle and COSMOS datasets.")
    test_X = pd.read_csv(cfg.TEST_FEATURES_PATH)
    return cosmos_X, cosmos_y, kaggle_X, kaggle_y, test_X
