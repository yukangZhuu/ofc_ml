"""Config-driven training pipeline.

Stages
------
- `run_pretrain(model, cfg, dataset_bundle)`   trains on COSMOS
- `run_finetune(model, cfg, dataset_bundle)`   trains on Kaggle
- `run_joint(model, cfg, dataset_bundle)`      trains on COSMOS ∪ Kaggle
- `orchestrate(cfg, dataset_bundle)`           main entry point; runs the
                                               stages listed in `cfg.stages`
                                               in order, reusing cached
                                               pretrain checkpoints when
                                               possible.

Each stage returns a `StageResult` with `best_loss`, `state_dict`, `history`.

Losses
------
- `MaskedMSELoss`
- `KaggleScoreLoss`  (kept for compatibility; selectable via stage.loss=='kaggle_score')

Compatibility
-------------
The pre-refactor `train_model_two_stage(...)` is preserved as a thin wrapper
over `orchestrate(...)` so `main.py` and `scripts/predict_finetuned.py` keep
working without modification.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from . import config as cfg_legacy
from .configs.schema import ExperimentConfig, StageConfig
from .features import preprocess_features
from .models import build_model, count_parameters
from .network import OFCDataset, compute_baseline_gain


CACHE_VERSION = "fno_zero_high_modes_gated_rng_fair_stage_seed_v4"


# ---------------------------------------------------------------------- #
# Device selection                                                       #
# ---------------------------------------------------------------------- #
def resolve_device(device_str: str = "auto") -> torch.device:
    ds = device_str.lower().strip()
    if ds in ("cpu",):
        return torch.device("cpu")
    if ds.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(ds)
        print(f"[device] Requested {ds} but CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    if ds == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------- #
# Losses                                                                 #
# ---------------------------------------------------------------------- #
class MaskedMSELoss(nn.Module):
    def forward(self, predictions, targets, mask):
        mask = mask.to(predictions.device)
        squared = (predictions - targets) ** 2
        masked = squared * mask
        denom = mask.sum().clamp_min(1.0)
        return masked.sum() / denom


class KaggleScoreLoss(nn.Module):
    """Differentiable approximation of the Kaggle score used for T95/Tmax-aware fine-tuning."""

    def __init__(self, tau: float = 0.5, beta: float = 0.7):
        super().__init__()
        self.tau = tau
        self.beta = beta

    def forward(self, predictions, targets, mask):
        diff = torch.abs(predictions - targets)
        e = diff * mask
        K = mask.sum(dim=1).clamp(min=1.0)
        a_i = e.sum(dim=1) / K
        a_bar = a_i.mean()
        e_centered = (e - a_i.unsqueeze(1)) * mask
        var_i = (e_centered ** 2).sum(dim=1) / K
        s_i = torch.sqrt(var_i + 1e-12)
        s_bar = s_i.mean()

        mask_bool = mask > 0
        all_err = e[mask_bool]
        if all_err.numel() == 0:
            return a_bar

        sorted_err, _ = torch.sort(all_err)
        n = sorted_err.numel()
        idx95 = int(0.95 * n)
        if idx95 < n:
            p95 = sorted_err[idx95:].mean()
        else:
            p95 = sorted_err[-1]
        t95 = F.relu(p95 - a_bar - self.tau)
        emax = all_err.max()
        t_max = F.relu(emax - p95 - self.beta)

        return a_bar + 0.3 * t95 + 0.1 * t_max + 0.15 * s_bar


def build_loss(name: str) -> nn.Module:
    n = name.lower().strip()
    if n == "masked_mse":
        return MaskedMSELoss()
    if n == "kaggle_score":
        return KaggleScoreLoss()
    raise ValueError(f"Unknown loss: {name!r}")


# ---------------------------------------------------------------------- #
# Data preparation                                                       #
# ---------------------------------------------------------------------- #
def prepare_offset_targets(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    preprocessor,
    mask_cols: List[str],
    target_cols: List[str],
    predict_absolute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run preprocessor and compute supervision targets.

    Returns
    -------
    X, y_target, target_gain, target_gain_tilt, mask
        `y_target` is either the offset (label - baseline, masked) or the raw
        label, depending on `predict_absolute`.
    """
    target_gain = features["target_gain"].values
    target_gain_tilt = features["target_gain_tilt"].values
    mask = features[mask_cols].values
    X = preprocessor.transform(features)
    y_raw = labels[target_cols].values
    if predict_absolute:
        y_target = y_raw * mask
    else:
        baseline = compute_baseline_gain(target_gain, target_gain_tilt)
        y_target = (y_raw - baseline) * mask
    return X, y_target, target_gain, target_gain_tilt, mask


def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    target_gain: np.ndarray,
    target_gain_tilt: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    val_size: float,
    random_state: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    n = len(X)
    if n < 2 or val_size is None or val_size <= 0.0:
        # Not enough samples for a held-out split; use everything for train,
        # and reuse the same set for val (only used for early-stop monitoring).
        tr = OFCDataset(X, y, target_gain, target_gain_tilt, mask)
        val = tr
    else:
        X_tr, X_val, y_tr, y_val, m_tr, m_val, tg_tr, tg_val, tgt_tr, tgt_val = train_test_split(
            X, y, mask, target_gain, target_gain_tilt,
            test_size=val_size, random_state=random_state,
        )
        tr = OFCDataset(X_tr, y_tr, tg_tr, tgt_tr, m_tr)
        val = OFCDataset(X_val, y_val, tg_val, tgt_val, m_val)

    nw = 4 if device.type == "cuda" else 0
    pin = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(int(random_state))
    train_loader = DataLoader(
        tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        generator=generator,
    )
    val_loader   = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pin)
    return train_loader, val_loader


# ---------------------------------------------------------------------- #
# Trainer                                                                #
# ---------------------------------------------------------------------- #
class Trainer:
    """A small training loop with validation, scheduler, and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        use_amp: bool = False,
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or MaskedMSELoss()
        self.scaler = GradScaler() if (use_amp and device.type == "cuda") else None
        self.grad_clip = float(grad_clip)

    def _train_epoch(self, loader: DataLoader, optimizer) -> float:
        self.model.train()
        total, denom = 0.0, 0.0
        for batch in loader:
            X, y, _, _, mask = [b.to(self.device, non_blocking=True) for b in batch]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.scaler is not None):
                preds = self.model(X, mask)
                loss = self.criterion(preds, y, mask)
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
            ms = float(mask.sum().item())
            total += float(loss.item()) * ms
            denom += ms
        return total / max(denom, 1.0)

    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total, denom = 0.0, 0.0
        with torch.no_grad():
            for batch in loader:
                X, y, _, _, mask = [b.to(self.device) for b in batch]
                with autocast(device_type=self.device.type, enabled=self.scaler is not None):
                    preds = self.model(X, mask)
                    loss = self.criterion(preds, y, mask)
                ms = float(mask.sum().item())
                total += float(loss.item()) * ms
                denom += ms
        return total / max(denom, 1.0)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        epochs: int,
        patience: int,
        scheduler=None,
        val_every_n: int = 1,
        title: str = "training",
    ) -> Tuple[float, List[Dict[str, float]]]:
        print(f"\n[{title}] epochs={epochs}, patience={patience}")
        best = float("inf")
        best_state = None
        bad = 0
        history: List[Dict[str, float]] = []

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer)
            if (epoch + 1) % val_every_n != 0:
                continue

            val_loss = self._validate(val_loader)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:4d}/{epochs} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.2e}")
            history.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss, "lr": lr})

            if val_loss < best:
                best = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stop at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best, history


def build_optimizer(model: nn.Module, stage: StageConfig) -> torch.optim.Optimizer:
    opt = stage.optimizer.lower()
    if opt == "adam":
        return optim.Adam(model.parameters(), lr=stage.learning_rate, weight_decay=stage.weight_decay)
    if opt == "adamw":
        return optim.AdamW(model.parameters(), lr=stage.learning_rate, weight_decay=stage.weight_decay)
    raise ValueError(f"Unknown optimizer: {stage.optimizer!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, stage: StageConfig):
    sch = stage.scheduler.lower()
    if sch == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7)
    raise ValueError(f"Unknown scheduler: {stage.scheduler!r}")


# ---------------------------------------------------------------------- #
# Stage results                                                          #
# ---------------------------------------------------------------------- #
@dataclass
class StageResult:
    stage: str
    best_loss: float
    state_dict: Dict[str, torch.Tensor]
    history: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class DatasetBundle:
    """Everything a stage needs, already preprocessed.

    Produced once per experiment by `prepare_dataset_bundle` so every stage
    shares the same preprocessor and the same per-row supervision scheme.
    """
    preprocessor: Any
    mask_cols: List[str]
    target_cols: List[str]
    input_dim: int
    output_dim: int
    # Train/eval tensors per dataset
    cosmos: Dict[str, np.ndarray]
    kaggle: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]
    # Raw frames kept for downstream evaluation/submission.
    test_features: pd.DataFrame
    test_labels: Optional[pd.DataFrame]


# ---------------------------------------------------------------------- #
# Pipeline orchestration                                                 #
# ---------------------------------------------------------------------- #
def prepare_dataset_bundle(
    cfg: ExperimentConfig,
    datasets,  # LoadedDatasets (forward-declared to avoid cyclic imports)
) -> DatasetBundle:
    """Fit the preprocessor on COSMOS ∪ Kaggle and pack tensors for each stage."""
    combined = pd.concat([datasets.cosmos_features, datasets.kaggle_features], axis=0, ignore_index=True)
    _, _, mask_cols, preprocessor = preprocess_features(combined, datasets.test_features)

    target_cols = sorted(
        [c for c in datasets.cosmos_labels.columns if "calculated_gain_spectra_" in c],
        key=lambda x: int(x.split("_")[-1]),
    )

    # Supervision targets
    predict_absolute = cfg.model.predict_absolute
    X_cos, y_cos, tg_cos, tgt_cos, m_cos = prepare_offset_targets(
        datasets.cosmos_features, datasets.cosmos_labels, preprocessor,
        mask_cols, target_cols, predict_absolute=predict_absolute,
    )
    X_kag, y_kag, tg_kag, tgt_kag, m_kag = prepare_offset_targets(
        datasets.kaggle_features, datasets.kaggle_labels, preprocessor,
        mask_cols, target_cols, predict_absolute=predict_absolute,
    )
    X_test = preprocessor.transform(datasets.test_features.drop(columns=["ID", "Usage"], errors="ignore"))

    input_dim = X_cos.shape[1]
    output_dim = y_cos.shape[1]

    return DatasetBundle(
        preprocessor=preprocessor,
        mask_cols=mask_cols,
        target_cols=target_cols,
        input_dim=input_dim,
        output_dim=output_dim,
        cosmos=dict(X=X_cos, y=y_cos, tg=tg_cos, tgt=tgt_cos, mask=m_cos),
        kaggle=dict(X=X_kag, y=y_kag, tg=tg_kag, tgt=tgt_kag, mask=m_kag),
        test=dict(
            X=X_test,
            tg=datasets.test_features["target_gain"].values,
            tgt=datasets.test_features["target_gain_tilt"].values,
            mask=datasets.test_features[mask_cols].values,
            ids=datasets.test_features["ID"].values,
        ),
        test_features=datasets.test_features,
        test_labels=datasets.test_labels,
    )


def _run_stage(
    model: nn.Module,
    device: torch.device,
    stage_cfg: StageConfig,
    data: Dict[str, np.ndarray],
    val_size: float,
    random_state: int,
    title: str,
) -> StageResult:
    if not stage_cfg.enabled:
        raise ValueError(f"Stage '{title}' invoked but stage_cfg.enabled is False")
    title_lower = title.lower()
    if "pretrain" in title_lower:
        stage_seed = int(random_state) + 1000
    elif "finetune" in title_lower:
        stage_seed = int(random_state) + 2000
    elif "joint" in title_lower:
        stage_seed = int(random_state) + 3000
    else:
        stage_seed = int(random_state)
    torch.manual_seed(stage_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(stage_seed)
    train_loader, val_loader = make_dataloaders(
        data["X"], data["y"], data["tg"], data["tgt"], data["mask"],
        batch_size=stage_cfg.batch_size,
        val_size=val_size,
        random_state=random_state,
        device=device,
    )
    criterion = build_loss(stage_cfg.loss)
    trainer = Trainer(model, device, criterion=criterion, grad_clip=stage_cfg.grad_clip)
    optimizer = build_optimizer(model, stage_cfg)
    scheduler = build_scheduler(optimizer, stage_cfg)
    best, history = trainer.fit(
        train_loader, val_loader, optimizer,
        epochs=stage_cfg.epochs,
        patience=stage_cfg.early_stopping_patience,
        scheduler=scheduler,
        val_every_n=stage_cfg.val_every_n_epochs,
        title=title,
    )
    return StageResult(
        stage=title,
        best_loss=best,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        history=history,
    )


def run_pretrain(model, device, cfg: ExperimentConfig, bundle: DatasetBundle) -> StageResult:
    return _run_stage(
        model, device, cfg.pretrain, bundle.cosmos,
        val_size=cfg.data.val_size, random_state=cfg.data.random_state,
        title="pretrain",
    )


def run_finetune(model, device, cfg: ExperimentConfig, bundle: DatasetBundle) -> StageResult:
    return _run_stage(
        model, device, cfg.finetune, bundle.kaggle,
        val_size=cfg.data.val_size, random_state=cfg.data.random_state,
        title="finetune",
    )


def run_joint(model, device, cfg: ExperimentConfig, bundle: DatasetBundle) -> StageResult:
    """Train on concat(COSMOS, Kaggle) as a single stage (A-T3)."""
    joined = {
        k: np.concatenate([bundle.cosmos[k], bundle.kaggle[k]], axis=0)
        for k in ("X", "y", "tg", "tgt", "mask")
    }
    return _run_stage(
        model, device, cfg.joint, joined,
        val_size=cfg.data.val_size, random_state=cfg.data.random_state,
        title="joint",
    )


# ---------------------------------------------------------------------- #
# Prediction helpers                                                     #
# ---------------------------------------------------------------------- #
def predict_test(
    model: nn.Module,
    device: torch.device,
    bundle: DatasetBundle,
    predict_absolute: bool = False,
) -> np.ndarray:
    """Return (N, 95) gain predictions aligned with bundle.test_features."""
    model.eval()
    X = torch.as_tensor(bundle.test["X"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(bundle.test["mask"], dtype=torch.float32, device=device)
    # In very small-memory situations we might want to chunk; test set is 21k rows
    # which is fine in one go on GPU.
    with torch.no_grad():
        pred = model(X, mask).cpu().numpy()
    if predict_absolute:
        out = pred
    else:
        baseline = compute_baseline_gain(bundle.test["tg"], bundle.test["tgt"])
        out = baseline + pred
    return out * bundle.test["mask"]


def save_submission(predictions: np.ndarray, bundle: DatasetBundle, output_path: Path) -> Path:
    target_cols = [f"calculated_gain_spectra_{i:02d}" for i in range(predictions.shape[1])]
    df = pd.DataFrame(predictions, columns=target_cols)
    df.insert(0, "ID", bundle.test["ids"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------- #
# Pretrain checkpoint cache                                              #
# ---------------------------------------------------------------------- #
def _arch_signature(cfg: ExperimentConfig, input_dim: int) -> str:
    """Hash that identifies a pretrain checkpoint's reusability.

    Two experiments with identical (model, pretrain stage config,
    cosmos_ratio, seed, input_dim, preprocessor setup) share the same
    pretrain outcome, so we can reuse a cached checkpoint.
    """
    key = {
        "cache_version": CACHE_VERSION,
        "model": cfg.model.__dict__,
        "pretrain": cfg.pretrain.__dict__,
        "cosmos_ratio": cfg.data.cosmos_ratio,
        "seed": cfg.seed,
        "use_mask": cfg.feature.use_mask,
        "input_dim": input_dim,
        "predict_absolute": cfg.model.predict_absolute,
    }
    blob = json.dumps(key, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _pretrain_cache_path(cfg: ExperimentConfig, signature: str) -> Path:
    cache_dir = Path(cfg.results_root) / "_pretrain_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cfg.model.name}_{signature}.pt"


# ---------------------------------------------------------------------- #
# Top-level orchestrator                                                 #
# ---------------------------------------------------------------------- #
def _load_weights_flexible(path: Path, model: nn.Module, device: torch.device) -> None:
    """Load weights from `path` into `model`, accepting several checkpoint layouts.

    Supported blobs on disk (all produced by the project):
      - plain state_dict
      - {"state_dict": ..., ...}           (new pretrain cache + pretrain.pt)
      - {"model_state_dict": ..., ...}     (legacy save_pretrained_model.pt)
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict")
        if sd is None:
            # assume the dict itself IS a state_dict (tensor values)
            sd = ckpt
    else:
        sd = ckpt
    model.load_state_dict(sd)


def orchestrate(
    cfg: ExperimentConfig,
    datasets,  # LoadedDatasets
    allow_pretrain_cache: bool = True,
) -> Dict[str, Any]:
    """Build model, run requested stages in order, and return a result dict.

    Behaviour summary:
      - If `cfg.pretrain_weights_path` is set, those weights are loaded into
        the model up front and any "pretrain" stage in `cfg.stages` is skipped
        (i.e. finetune resumes from those weights).
      - Otherwise the "pretrain" stage (if present) runs normally, optionally
        hitting the cross-experiment hash cache at `results/_pretrain_cache/`.
      - Whenever a pretrain stage finishes successfully, the resulting weights
        are written both to the cache and to `<results_dir>/pretrain.pt` for
        easy per-experiment reuse.

    Returns
    -------
    dict with keys:
      - 'model'            : the trained torch.nn.Module (on `device`)
      - 'device'           : torch.device
      - 'bundle'           : DatasetBundle (preprocessor, tensors, test frames)
      - 'stage_results'    : List[StageResult]
      - 'predict_absolute' : bool (copied from cfg.model for downstream use)
      - 'pretrain_path'    : Optional[Path] of the per-experiment pretrain.pt
    """
    device = resolve_device(cfg.device)
    print(f"[orchestrate] device = {device}")

    # 1. Dataset bundle
    bundle = prepare_dataset_bundle(cfg, datasets)
    print(f"[orchestrate] input_dim={bundle.input_dim}, output_dim={bundle.output_dim}")

    # 2. Model
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_model(cfg.model, input_dim=bundle.input_dim, output_dim=bundle.output_dim).to(device)
    print(f"[orchestrate] model={cfg.model.name} params={count_parameters(model):,}")
    # Model construction can consume a different amount of RNG depending on
    # architecture (e.g. enabling SpectralMixing creates extra parameters).
    # Reset the training RNG after construction so dropout and other training
    # stochasticity are comparable across same-seed ablations.
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    stage_results: List[StageResult] = []
    arch_sig = _arch_signature(cfg, bundle.input_dim)
    cfg.pretrain_cache_key = arch_sig
    cache_path = _pretrain_cache_path(cfg, arch_sig)

    # 3. Optional: resume-from-pretrain (explicit user-supplied weights).
    results_dir = Path(cfg.results_dir)
    per_exp_pretrain_path = results_dir / "pretrain.pt"
    pretrain_resumed = False
    if cfg.pretrain_weights_path:
        weights_path = Path(cfg.pretrain_weights_path)
        if not weights_path.is_absolute():
            weights_path = (Path.cwd() / weights_path).resolve()
        print(f"[orchestrate] resume: loading pretrain weights from {weights_path}")
        _load_weights_flexible(weights_path, model, device)
        pretrain_resumed = True
        # Record a synthetic StageResult so downstream bookkeeping still sees a pretrain entry.
        stage_results.append(
            StageResult(
                stage="pretrain",
                best_loss=float("nan"),
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                history=[{"epoch": 0, "train": float("nan"), "val": float("nan"), "lr": float("nan"),
                          "note": f"resumed from {weights_path}"}],
            )
        )

    # 4. Run stages in order
    for stage in cfg.stages:
        s = stage.lower().strip()
        if s == "pretrain":
            if pretrain_resumed:
                print("[orchestrate] skipping 'pretrain' stage because pretrain_weights_path was provided.")
                continue
            if allow_pretrain_cache and cache_path.exists():
                print(f"[orchestrate] reusing pretrain cache: {cache_path}")
                ckpt = torch.load(cache_path, map_location=device)
                model.load_state_dict(ckpt["state_dict"])
                stage_results.append(
                    StageResult(
                        stage="pretrain",
                        best_loss=float(ckpt.get("best_loss", float("nan"))),
                        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        history=ckpt.get("history", []),
                    )
                )
            else:
                res = run_pretrain(model, device, cfg, bundle)
                stage_results.append(res)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "best_loss": res.best_loss,
                        "history": res.history,
                        "arch_sig": arch_sig,
                        "cache_version": CACHE_VERSION,
                        "model_name": cfg.model.name,
                    },
                    cache_path,
                )
                print(f"[orchestrate] pretrain cached at {cache_path}")

            # Always snapshot the pretrain weights per-experiment (even when
            # reusing the cache) so users can cherry-pick an experiment's
            # pretrain state as the starting point for another run.
            per_exp_pretrain_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_loss": stage_results[-1].best_loss,
                    "history": stage_results[-1].history,
                    "arch_sig": arch_sig,
                    "cache_version": CACHE_VERSION,
                    "model_name": cfg.model.name,
                    "experiment_name": cfg.name,
                    "saved_by": "orchestrate/pretrain",
                },
                per_exp_pretrain_path,
            )
            print(f"[orchestrate] per-experiment pretrain snapshot at {per_exp_pretrain_path}")
        elif s == "finetune":
            res = run_finetune(model, device, cfg, bundle)
            stage_results.append(res)
        elif s == "joint":
            res = run_joint(model, device, cfg, bundle)
            stage_results.append(res)
        else:
            raise ValueError(f"Unknown stage: {stage!r} (expected: pretrain, finetune, joint)")

    return {
        "model": model,
        "device": device,
        "bundle": bundle,
        "stage_results": stage_results,
        "predict_absolute": cfg.model.predict_absolute,
        "pretrain_path": per_exp_pretrain_path if per_exp_pretrain_path.exists() else None,
    }


# ---------------------------------------------------------------------- #
# Legacy wrapper (back-compat with pre-refactor entry points)            #
# ---------------------------------------------------------------------- #
class PyTorchModelWrapper:
    """Mimics the pre-refactor wrapper used by `scripts/predict_finetuned.py`."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, X, target_gain, target_gain_tilt, mask=None):
        self.model.eval()
        with torch.no_grad():
            tX = torch.FloatTensor(X).to(self.device)
            if mask is not None:
                tM = torch.FloatTensor(mask).to(self.device)
                preds_off = self.model(tX, tM)
            else:
                preds_off = self.model(tX)
            preds_off = preds_off.cpu().numpy()
            baseline = compute_baseline_gain(target_gain, target_gain_tilt)
            out = baseline + preds_off
            if mask is not None:
                out = out * mask
            return out


def train_model_two_stage(
    cosmos_features, cosmos_labels,
    kaggle_features, kaggle_labels,
    test_features,
    preprocessor,
    mask_cols,
):
    """Legacy entry point used by `main.py` and `scripts/predict_finetuned.py`.

    Reads runtime flags from the legacy `config.py` globals so existing
    callers keep working.  For new experiments use `orchestrate(cfg, ...)`
    directly.
    """
    from .data import LoadedDatasets  # local import to avoid cycle

    device = resolve_device(getattr(cfg_legacy, "DEVICE", "auto"))
    print(f"[legacy train_model_two_stage] device={device}")

    target_cols = sorted(
        [c for c in cosmos_labels.columns if "calculated_gain_spectra_" in c],
        key=lambda x: int(x.split("_")[-1]),
    )

    predict_absolute = False
    X_cos, y_cos, tg_cos, tgt_cos, m_cos = prepare_offset_targets(
        cosmos_features, cosmos_labels, preprocessor, mask_cols, target_cols, predict_absolute,
    )
    X_kag, y_kag, tg_kag, tgt_kag, m_kag = prepare_offset_targets(
        kaggle_features, kaggle_labels, preprocessor, mask_cols, target_cols, predict_absolute,
    )

    torch.manual_seed(getattr(cfg_legacy, "RANDOM_STATE", 42))
    np.random.seed(getattr(cfg_legacy, "RANDOM_STATE", 42))
    # Build model via the registry using legacy config values.
    from .configs.schema import ModelConfig
    mcfg = ModelConfig(
        name="hybrid_fno_kan",
        hidden_dims=list(cfg_legacy.HYBRID_FNO_KAN_HIDDEN_DIMS),
        dropout=cfg_legacy.HYBRID_FNO_KAN_DROPOUT,
        n_frequencies=cfg_legacy.HYBRID_FNO_KAN_N_FREQUENCIES,
        spectral_freq_ratio=cfg_legacy.HYBRID_FNO_KAN_SPECTRAL_FREQ_RATIO,
        use_spectral_mixing=cfg_legacy.HYBRID_FNO_KAN_USE_SPECTRAL_MIXING,
    )
    model = build_model(mcfg, input_dim=X_cos.shape[1], output_dim=y_cos.shape[1]).to(device)
    print(f"[legacy] params={count_parameters(model):,}")

    # Stage 1 (pretrain) with optional load
    pretrain_loss = None
    pretrain_path = Path(cfg_legacy.PRETRAIN_MODEL_PATH)
    ran_pretrain = True
    if getattr(cfg_legacy, "LOAD_PRETRAINED_MODEL", False) and pretrain_path.exists():
        try:
            ckpt = torch.load(pretrain_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            pretrain_loss = ckpt.get("pretrain_loss")
            ran_pretrain = False
            print(f"[legacy] loaded pretrained weights from {pretrain_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[legacy] failed to load pretrained weights: {e}; will retrain")

    if ran_pretrain:
        stage = StageConfig(
            learning_rate=cfg_legacy.PRETRAIN_LEARNING_RATE,
            weight_decay=cfg_legacy.PRETRAIN_WEIGHT_DECAY,
            batch_size=cfg_legacy.PRETRAIN_BATCH_SIZE,
            epochs=cfg_legacy.PRETRAIN_EPOCHS,
            early_stopping_patience=cfg_legacy.PRETRAIN_EARLY_STOPPING_PATIENCE,
            val_every_n_epochs=cfg_legacy.PRETRAIN_VAL_EVERY_N_EPOCHS,
            optimizer="adam",
            loss="masked_mse",
        )
        res = _run_stage(
            model, device, stage,
            dict(X=X_cos, y=y_cos, tg=tg_cos, tgt=tgt_cos, mask=m_cos),
            val_size=getattr(cfg_legacy, "TEST_SIZE", 0.05),
            random_state=getattr(cfg_legacy, "RANDOM_STATE", 42),
            title="legacy pretrain",
        )
        pretrain_loss = res.best_loss
        pretrain_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "pretrain_loss": pretrain_loss,
            "input_dim": X_cos.shape[1],
            "output_dim": y_cos.shape[1],
        }, pretrain_path)

    pretrained_state = copy.deepcopy(model.state_dict())

    # Stage 2 (finetune)
    use_kaggle_loss = str(getattr(cfg_legacy, "USE_KAGGLE_SCORE_LOSS", "none")).lower() in {"finetune", "both"}
    ft_stage = StageConfig(
        learning_rate=cfg_legacy.FINETUNE_LEARNING_RATE,
        weight_decay=cfg_legacy.FINETUNE_WEIGHT_DECAY,
        batch_size=cfg_legacy.FINETUNE_BATCH_SIZE,
        epochs=cfg_legacy.FINETUNE_EPOCHS,
        early_stopping_patience=cfg_legacy.FINETUNE_EARLY_STOPPING_PATIENCE,
        val_every_n_epochs=cfg_legacy.FINETUNE_VAL_EVERY_N_EPOCHS,
        optimizer="adamw",
        loss="kaggle_score" if use_kaggle_loss else "masked_mse",
    )
    ft_res = _run_stage(
        model, device, ft_stage,
        dict(X=X_kag, y=y_kag, tg=tg_kag, tgt=tgt_kag, mask=m_kag),
        val_size=getattr(cfg_legacy, "TEST_SIZE", 0.05),
        random_state=getattr(cfg_legacy, "RANDOM_STATE", 42),
        title="legacy finetune",
    )

    ft_path = Path(cfg_legacy.FINETUNE_MODEL_PATH)
    ft_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "finetune_loss": ft_res.best_loss,
        "input_dim": X_kag.shape[1],
        "output_dim": y_kag.shape[1],
    }, ft_path)

    # Evaluation mirroring the original code
    X_k_val = X_kag  # legacy split is recomputed inside fit; we just report on full kaggle
    mask_k_val = m_kag
    y_offset_k_val = y_kag
    with torch.no_grad():
        y_pred_off = model(
            torch.FloatTensor(X_k_val).to(device),
            torch.FloatTensor(mask_k_val).to(device),
        ).cpu().numpy()
    non_zero = mask_k_val > 0
    mse = mean_squared_error((y_offset_k_val * mask_k_val)[non_zero], (y_pred_off * mask_k_val)[non_zero])
    mae = mean_absolute_error((y_offset_k_val * mask_k_val)[non_zero], (y_pred_off * mask_k_val)[non_zero])
    rmse = float(np.sqrt(mse))
    print(f"[legacy] Kaggle-train MSE={mse:.6f} RMSE={rmse:.6f} MAE={mae:.6f}")

    wrapper_ft = PyTorchModelWrapper(model, device)
    # Pretrained wrapper (re-built from saved state)
    from .configs.schema import ModelConfig as _MC
    mcfg2 = mcfg
    model_pt = build_model(mcfg2, input_dim=X_cos.shape[1], output_dim=y_cos.shape[1]).to(device)
    model_pt.load_state_dict(pretrained_state)
    wrapper_pt = PyTorchModelWrapper(model_pt, device)

    metrics = {
        "mse": float(mse), "mae": float(mae), "rmse": float(rmse),
        "pretrain_loss": float(pretrain_loss) if pretrain_loss is not None else None,
        "finetune_loss": float(ft_res.best_loss),
    }
    return wrapper_ft, wrapper_pt, metrics
