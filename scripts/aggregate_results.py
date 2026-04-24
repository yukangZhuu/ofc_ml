#!/usr/bin/env python3
"""Collect per-experiment metrics.json files into paper-ready tables.

Outputs (written to `results/_tables/`):
    table1_main.csv            # §2.1 Main results (ours + 3 baselines)
    table2_transfer.csv        # §2.2 Transfer-learning ablation
    table3_arch.csv            # §2.3 Architectural ablation
    figure3_data_scale.csv     # §2.4 Data-scale ablation (long format)

Each CSV row is one experiment; columns cover the headline metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_ROOT / "_tables"

MAIN_EXPS = ["m1_ours", "m2_mlp", "m3_cnn1d", "m4_transformer"]
TRANSFER_EXPS = ["m1_ours", "a_t1_no_finetune", "a_t2_no_pretrain", "a_t3_joint"]
ARCH_EXPS = ["m1_ours", "a_a1_wo_spectral", "a_a2_wo_fourier_kan", "m2_mlp"]
PHYSICS_EXPS = ["m1_ours", "a_p1_predict_absolute"]
DATA_SCALE_EXPS = [
    "ds_pretrain_25", "ds_pretrain_50", "ds_pretrain_100",
    "ds_finetune_25", "ds_finetune_50", "ds_finetune_100",
]


def _load_metrics(name: str) -> Optional[Dict[str, Any]]:
    p = RESULTS_ROOT / name / "metrics.json"
    if not p.exists():
        print(f"[aggregate] missing metrics.json for {name}; skipping.")
        return None
    with p.open("r") as f:
        return json.load(f)


def _row_from_metrics(name: str, m: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a metrics.json into a single-row dict with canonical columns.

    All MAE/RMSE/Std/T95/Tmax/KaggleScore values are in **dB**; MSE is in
    **dB^2**.  Column names carry an explicit `_dB` suffix to avoid any
    ambiguity in downstream paper tables.
    """
    ev = m.get("evaluation", {}) or {}
    ov = ev.get("overall", {})
    by_u = ev.get("by_usage", {})
    by_c = ev.get("by_category", {})
    by_et = ev.get("by_edfa_type", {})

    def g(d, k):
        return d.get(k, float("nan")) if isinstance(d, dict) else float("nan")

    return {
        "experiment": name,
        "model": m.get("model_name"),
        "stages": ",".join(m.get("stages", []) or []),
        "params": m.get("params"),
        "cosmos_ratio": m.get("cosmos_ratio"),
        "kaggle_ratio": m.get("kaggle_ratio"),
        "predict_absolute": m.get("predict_absolute"),
        # Overall  (dB / dB^2)
        "MAE_dB":           g(ov, "MAE"),
        "RMSE_dB":          g(ov, "RMSE"),
        "MSE_dB2":          g(ov, "MSE"),
        "Std_dB":           g(ov, "Std"),
        "T95_dB":           g(ov, "T95"),
        "Tmax_dB":          g(ov, "Tmax"),
        "KaggleScore_dB":   g(ov, "KaggleScore"),
        # Public / Private (dB)
        "Public_MAE_dB":    g(by_u.get("Public", {}), "MAE"),
        "Public_Score_dB":  g(by_u.get("Public", {}), "KaggleScore"),
        "Private_MAE_dB":   g(by_u.get("Private", {}), "MAE"),
        "Private_Score_dB": g(by_u.get("Private", {}), "KaggleScore"),
        # Category breakdowns (dB)
        "aging_MAE_dB":     g(by_c.get("aging", {}), "MAE"),
        "shb_MAE_dB":       g(by_c.get("shb", {}), "MAE"),
        "unseen_MAE_dB":    g(by_c.get("unseen", {}), "MAE"),
        "cosmos_MAE_dB":    g(by_c.get("cosmos", {}), "MAE"),
        # EDFA type (dB)
        "booster_MAE_dB":   g(by_et.get("booster", {}), "MAE"),
        "preamp_MAE_dB":    g(by_et.get("preamp", {}), "MAE"),
        # Stage losses (unitless: masked MSE of target space used by that stage)
        "pretrain_loss": (m.get("stage_losses", {}) or {}).get("pretrain"),
        "finetune_loss": (m.get("stage_losses", {}) or {}).get("finetune"),
        "joint_loss":    (m.get("stage_losses", {}) or {}).get("joint"),
        "wall_time_sec": m.get("wall_time_sec"),
    }


def _table_for(names: List[str]) -> pd.DataFrame:
    rows = []
    for n in names:
        m = _load_metrics(n)
        if m is None:
            continue
        rows.append(_row_from_metrics(n, m))
    return pd.DataFrame(rows)


def _data_scale_long(names: List[str]) -> pd.DataFrame:
    """Long-format table for figure 3.

    Columns: experiment, axis (pretrain|finetune), ratio, MAE, KaggleScore.
    """
    rows = []
    for n in names:
        m = _load_metrics(n)
        if m is None:
            continue
        row = _row_from_metrics(n, m)
        # Determine axis from naming convention
        if n.startswith("ds_pretrain_"):
            axis = "pretrain"
            ratio = row["cosmos_ratio"]
        elif n.startswith("ds_finetune_"):
            axis = "finetune"
            ratio = row["kaggle_ratio"]
        else:
            axis = "unknown"
            ratio = None
        rows.append(
            {
                "experiment": n,
                "axis": axis,
                "ratio": ratio,
                "cosmos_ratio": row["cosmos_ratio"],
                "kaggle_ratio": row["kaggle_ratio"],
                "MAE_dB": row["MAE_dB"],
                "RMSE_dB": row["RMSE_dB"],
                "KaggleScore_dB": row["KaggleScore_dB"],
                "Public_Score_dB": row["Public_Score_dB"],
                "Private_Score_dB": row["Private_Score_dB"],
            }
        )
    return pd.DataFrame(rows)


def main():
    global RESULTS_ROOT, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(RESULTS_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    RESULTS_ROOT = Path(args.results_root)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "table1_main.csv":           _table_for(MAIN_EXPS),
        "table2_transfer.csv":       _table_for(TRANSFER_EXPS),
        "table3_arch.csv":           _table_for(ARCH_EXPS),
        "table4_physics.csv":        _table_for(PHYSICS_EXPS),
        "figure3_data_scale.csv":    _data_scale_long(DATA_SCALE_EXPS),
    }
    for name, df in tables.items():
        out = OUT_DIR / name
        df.to_csv(out, index=False)
        print(f"[aggregate] {out} ({len(df)} rows) -- all dB columns end in _dB; MSE in dB^2")
        if len(df):
            cols = [c for c in
                    ["experiment", "MAE_dB", "RMSE_dB", "KaggleScore_dB",
                     "Public_Score_dB", "Private_Score_dB", "ratio"]
                    if c in df.columns]
            print(df[cols].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
