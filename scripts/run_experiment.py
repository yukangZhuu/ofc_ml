#!/usr/bin/env python3
"""Run a single experiment described by a YAML config.

Usage
-----
    python scripts/run_experiment.py --config experiments/main/m1_ours.yaml
    python scripts/run_experiment.py --config experiments/ablation/arch/a_a1_wo_spectral.yaml
                                     --override device=cuda:0 finetune.epochs=50

Outputs
-------
Everything lands under `results/<exp_name>/`:
    metrics.json             # compute_metrics + stage losses + param count
    submission.csv           # aligned to test_features ID
    model.pt                 # final trained weights
    config.snapshot.yaml     # resolved config (base + override)
    logs/train.log           # redirected stdout
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

# Silence MPS FFT resize warnings that spam the log on Apple Silicon.
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ofc_ml.configs import load_experiment_config  # noqa: E402
from ofc_ml.data import load_datasets  # noqa: E402
from ofc_ml.evaluate import compute_metrics  # noqa: E402
from ofc_ml.model import orchestrate, predict_test, save_submission  # noqa: E402
from ofc_ml.models import count_parameters  # noqa: E402


def _apply_overrides(cfg, overrides: list[str]) -> None:
    """Apply `key.subkey=value` style CLI overrides to the config."""
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        # Best-effort YAML parsing (handles bool/int/float/list)
        val = yaml.safe_load(raw)
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)


def _exists_already(results_dir: Path) -> bool:
    return (results_dir / "metrics.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment YAML")
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--force", action="store_true", help="Re-run even if metrics.json exists")
    ap.add_argument(
        "--no-cache", action="store_true",
        help="Disable pretrain checkpoint cache (force pretrain to actually run)",
    )
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    _apply_overrides(cfg, args.override)

    results_dir = Path(cfg.results_dir)
    if _exists_already(results_dir) and not args.force:
        print(f"[runner] {cfg.name}: metrics.json already exists at {results_dir}; skipping.")
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)

    # Snapshot config
    with (results_dir / "config.snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg.as_serializable(), f, sort_keys=False)
    shutil.copy(args.config, results_dir / "config.source.yaml")

    print(f"\n=== Running experiment: {cfg.name} ===")
    print(f"    stages      : {cfg.stages}")
    print(f"    model       : {cfg.model.name}")
    print(f"    cosmos_ratio: {cfg.data.cosmos_ratio}")
    print(f"    kaggle_ratio: {cfg.data.kaggle_ratio}")
    print(f"    device      : {cfg.device}")

    t0 = time.time()
    datasets = load_datasets(cfg)
    out = orchestrate(cfg, datasets, allow_pretrain_cache=not args.no_cache)
    model = out["model"]
    device = out["device"]
    bundle = out["bundle"]

    # Save final weights
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.as_serializable()}, results_dir / "model.pt")

    # Persist per-epoch training history (both in metrics.json and as a flat
    # CSV that is much easier to plot from notebooks).
    import csv
    history_csv = results_dir / "history.csv"
    stage_histories = {}
    with history_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stage", "epoch", "train", "val", "lr", "note"])
        for r in out["stage_results"]:
            stage_histories[r.stage] = r.history
            for row in r.history:
                writer.writerow([
                    r.stage,
                    row.get("epoch"),
                    row.get("train"),
                    row.get("val"),
                    row.get("lr"),
                    row.get("note", ""),
                ])
    print(f"[runner] per-epoch history at {history_csv}")

    # Predict + submission
    preds = predict_test(model, device, bundle, predict_absolute=cfg.model.predict_absolute)
    sub_path = save_submission(preds, bundle, results_dir / "submission.csv")
    print(f"[runner] submission saved to {sub_path}")

    # Evaluate
    metrics = {
        "experiment_name": cfg.name,
        "model_name": cfg.model.name,
        "stages": cfg.stages,
        "params": count_parameters(model),
        "cosmos_ratio": cfg.data.cosmos_ratio,
        "kaggle_ratio": cfg.data.kaggle_ratio,
        "predict_absolute": bool(cfg.model.predict_absolute),
        "pretrain_weights_path": cfg.pretrain_weights_path,
        "stage_losses": {r.stage: float(r.best_loss) for r in out["stage_results"]},
        "stage_histories": {
            k: [{kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                 for kk, vv in row.items()}
                for row in v]
            for k, v in stage_histories.items()
        },
        "wall_time_sec": float(time.time() - t0),
    }
    if bundle.test_labels is not None:
        result = compute_metrics(
            preds,
            bundle.test_labels,
            bundle.test_features,
            per_usage=cfg.eval.per_usage,
            per_category=cfg.eval.per_category,
            per_edfa_type=cfg.eval.per_edfa_type,
        )
        metrics["evaluation"] = result.as_dict()
    else:
        print("[runner] test_labels not available; skipping offline evaluation.")

    with (results_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"[runner] metrics written to {results_dir / 'metrics.json'}")
    if "evaluation" in metrics:
        ov = metrics["evaluation"]["overall"]
        print(
            f"[runner] OVERALL  MAE={ov['MAE']:.4f}  RMSE={ov['RMSE']:.4f}  KaggleScore={ov['KaggleScore']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
