"""
YAML loader for ExperimentConfig.

We support a light "overlay" pattern: every experiment YAML may set a
`base: <path>` field whose referenced YAML is loaded first and then overridden
by the current file's fields (recursively, dict-merged).  Paths are resolved
relative to the including file.

Usage
-----
    from ofc_ml.configs import load_experiment_config
    cfg = load_experiment_config("experiments/main/m1_ours.yaml")
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .. import config as legacy_cfg
from .schema import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    StageConfig,
    WangTransferConfig,
)


PROJECT_ROOT = legacy_cfg.PROJECT_ROOT


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `override` into `base` recursively.  Lists are replaced whole."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml_with_base(path: Path) -> Dict[str, Any]:
    """Read a YAML file and resolve its `base:` chain (if any)."""
    path = path.resolve()
    with path.open("r") as f:
        data: Dict[str, Any] = yaml.safe_load(f) or {}

    base_ref = data.pop("base", None)
    if base_ref is None:
        return data

    base_path = (path.parent / base_ref).resolve()
    base_data = _read_yaml_with_base(base_path)
    return _deep_merge(base_data, data)


# ---------------------------------------------------------------------- #
# dict -> dataclass conversion                                           #
# ---------------------------------------------------------------------- #

_PATH_FIELDS = {
    "data_dir",
    "train_features_path",
    "train_labels_path",
    "test_features_path",
    "test_labels_path",
    "cosmos_data_dir",
    "cosmos_train_features_path",
    "cosmos_train_labels_path",
    "results_root",
}


def _resolve_path(v: Any) -> Path:
    if isinstance(v, Path):
        return v
    p = Path(str(v))
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _coerce_dataclass(cls, payload: Dict[str, Any]):
    """Recursively convert a dict into the given dataclass `cls`, falling back
    to dataclass defaults for fields not present in the payload.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    kwargs: Dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload.keys()) - set(known.keys())
    if unknown:
        raise ValueError(
            f"Unknown config field(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Allowed: {sorted(known.keys())}"
        )
    for fname, fspec in known.items():
        if fname not in payload:
            continue
        v = payload[fname]
        ftype = fspec.type
        # Nested dataclasses
        if isinstance(v, dict) and ftype in {
            DataConfig,
            FeatureConfig,
            ModelConfig,
            StageConfig,
            EvalConfig,
            WangTransferConfig,
        }:
            kwargs[fname] = _coerce_dataclass(ftype, v)
        elif fname in _PATH_FIELDS:
            kwargs[fname] = _resolve_path(v)
        else:
            kwargs[fname] = v
    return cls(**kwargs)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML (with optional `base:`) into ExperimentConfig."""
    p = Path(path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Experiment config not found: {p}")

    data = _read_yaml_with_base(p)

    if "name" not in data:
        # Default the experiment name to the YAML filename stem.
        data["name"] = p.stem

    # Type hints on dataclass fields are stored as strings when `from __future__
    # import annotations` is active, so we rewrite to actual classes here.
    _TYPE_MAP = {
        "DataConfig": DataConfig,
        "FeatureConfig": FeatureConfig,
        "ModelConfig": ModelConfig,
        "StageConfig": StageConfig,
        "EvalConfig": EvalConfig,
        "WangTransferConfig": WangTransferConfig,
    }
    for f in fields(ExperimentConfig):
        if isinstance(f.type, str) and f.type in _TYPE_MAP:
            f.type = _TYPE_MAP[f.type]
    for dc in (DataConfig, FeatureConfig, ModelConfig, StageConfig, EvalConfig, WangTransferConfig):
        for f in fields(dc):
            if isinstance(f.type, str) and f.type in _TYPE_MAP:
                f.type = _TYPE_MAP[f.type]

    return _coerce_dataclass(ExperimentConfig, data)
