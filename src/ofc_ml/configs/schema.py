"""
ExperimentConfig schema (dataclass-based).

A single experiment is fully specified by an `ExperimentConfig` instance.
YAML files in `experiments/` are merged on top of `base.yaml` and then loaded
into this schema.  All downstream modules (data loading, feature preprocessing,
model building, training, evaluation) read from an `ExperimentConfig` object,
not from module-level globals in `config.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config as legacy_cfg


@dataclass
class DataConfig:
    """Paths + dataset subsampling for a single experiment."""
    data_dir: Path = field(default_factory=lambda: legacy_cfg.DATA_DIR)
    train_features_path: Path = field(default_factory=lambda: legacy_cfg.TRAIN_FEATURES_PATH)
    train_labels_path: Path = field(default_factory=lambda: legacy_cfg.TRAIN_LABELS_PATH)
    test_features_path: Path = field(default_factory=lambda: legacy_cfg.TEST_FEATURES_PATH)
    test_labels_path: Path = field(
        default_factory=lambda: legacy_cfg.DATA_DIR / "test_labels.csv"
    )
    cosmos_data_dir: Path = field(default_factory=lambda: legacy_cfg.COSMOS_DATA_DIR)
    cosmos_train_features_path: Path = field(
        default_factory=lambda: legacy_cfg.COSMOS_TRAIN_FEATURES_PATH
    )
    cosmos_train_labels_path: Path = field(
        default_factory=lambda: legacy_cfg.COSMOS_TRAIN_LABELS_PATH
    )
    # Subsampling ratios (used in data-scale ablation).  1.0 = use all rows.
    cosmos_ratio: float = 1.0
    kaggle_ratio: float = 1.0
    # train/val split on the (optionally subsampled) fine-tuning dataset.
    val_size: float = 0.05
    random_state: int = 42


@dataclass
class FeatureConfig:
    """Feature preprocessing settings."""
    use_mask: str = "concat"  # one of {"none", "concat", "multiply"}


@dataclass
class ModelConfig:
    """Architecture-level hyperparameters.

    The `name` field drives `build_model` in `src/ofc_ml/models/__init__.py`:
      - "hybrid_fno_kan": current HybridFNOKANPredictor (with boolean switches)
      - "mlp":            same-size MLP baseline
      - "cnn1d":          1D Conv baseline (channel-as-sequence)
      - "transformer":    Transformer encoder baseline (channel-as-token)

    Fields not relevant to a given architecture are silently ignored.
    """
    name: str = "hybrid_fno_kan"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128, 128, 128])
    dropout: float = 0.2
    # HybridFNOKAN specific
    n_frequencies: int = 4
    spectral_freq_ratio: float = 0.5
    use_spectral_mixing: bool = True
    use_fourier_kan: bool = True
    use_residual: bool = True
    # Predict-residual (baseline subtraction) switch; kept False -> predict offset.
    # If True, model predicts absolute gain directly (no baseline subtraction).
    predict_absolute: bool = False
    # CNN1D specific (tuned so total params ~ HybridFNOKAN default)
    cnn_channels: List[int] = field(default_factory=lambda: [64, 96, 128, 160])
    cnn_kernel_size: int = 7
    # Transformer specific (tuned so total params ~ HybridFNOKAN default)
    tr_d_model: int = 96
    tr_nhead: int = 4
    tr_num_layers: int = 4
    tr_dim_feedforward: int = 192


@dataclass
class StageConfig:
    """Training config for a single stage (pretrain, finetune, joint)."""
    enabled: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 500
    early_stopping_patience: int = 40
    val_every_n_epochs: int = 2
    optimizer: str = "adam"  # "adam" or "adamw"
    loss: str = "masked_mse"  # "masked_mse" or "kaggle_score"
    scheduler: str = "reduce_on_plateau"  # only option now
    grad_clip: float = 1.0


@dataclass
class EvalConfig:
    """Evaluation config (offline, on test_labels.csv)."""
    save_submission: bool = True
    # Fine-grained breakdowns to compute.
    per_category: bool = True
    per_edfa_type: bool = True
    # Public/Private split via Usage column.
    per_usage: bool = True


@dataclass
class ExperimentConfig:
    """A fully-specified experiment."""
    name: str
    # Stages to execute in order.  Supported values:
    #   "pretrain", "finetune", "joint"
    # Examples:
    #   ["pretrain", "finetune"]  -> main protocol (M-1..M-4)
    #   ["pretrain"]              -> A-T1 No-Finetune
    #   ["finetune"]              -> A-T2 No-Pretrain (finetune acts on Kaggle from scratch)
    #   ["joint"]                 -> A-T3 merged single-stage training on COSMOS ∪ Kaggle
    stages: List[str] = field(default_factory=lambda: ["pretrain", "finetune"])
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", "mps"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: StageConfig = field(
        default_factory=lambda: StageConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            epochs=500,
            early_stopping_patience=40,
            val_every_n_epochs=2,
            optimizer="adam",
        )
    )
    finetune: StageConfig = field(
        default_factory=lambda: StageConfig(
            learning_rate=2e-4,
            weight_decay=5e-5,
            batch_size=64,
            epochs=500,
            early_stopping_patience=60,
            val_every_n_epochs=1,
            optimizer="adamw",
        )
    )
    # Optional joint-training stage (used by A-T3 Joint). When stages==["joint"],
    # we use `joint` settings as the single training stage over COSMOS ∪ Kaggle.
    joint: StageConfig = field(
        default_factory=lambda: StageConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            epochs=500,
            early_stopping_patience=40,
            val_every_n_epochs=2,
            optimizer="adam",
        )
    )
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ #
    # Output layout (filled in by the runner, not typically by YAML)     #
    # ------------------------------------------------------------------ #
    results_root: Path = field(default_factory=lambda: Path("results"))
    # When set, the pretrain stage will try to reuse a cached checkpoint
    # keyed by `pretrain_cache_key`.  The runner computes this key from
    # model/arch/data hash.
    pretrain_cache_key: Optional[str] = None
    # When set (either absolute path or repo-relative), these weights are
    # loaded into the model before any stage runs and the "pretrain" stage
    # is automatically skipped (so the subsequent stages, typically
    # "finetune", start from these weights).  Accepts a plain torch
    # state_dict, or a checkpoint dict containing one of {state_dict,
    # model_state_dict}.
    pretrain_weights_path: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    @property
    def results_dir(self) -> Path:
        return self.results_root / self.name

    def as_serializable(self) -> Dict[str, Any]:
        """Return a plain nested dict suitable for YAML/JSON dumping."""

        def _convert(v: Any) -> Any:
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, (StageConfig, DataConfig, FeatureConfig, ModelConfig, EvalConfig, ExperimentConfig)):
                return {k: _convert(getattr(v, k)) for k in v.__dataclass_fields__}
            if isinstance(v, list):
                return [_convert(x) for x in v]
            if isinstance(v, dict):
                return {k: _convert(x) for k, x in v.items()}
            return v

        return _convert(self)
