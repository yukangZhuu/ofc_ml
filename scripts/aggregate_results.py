#!/usr/bin/env python3
"""Aggregate per-experiment metrics.json files across seeds into paper tables.

New layout (post multi-seed refactor)
-------------------------------------
Per-seed outputs live under `results/seed_<SEED>/<experiment>/metrics.json`.

For every paper table this script emits **two** CSVs to `results/_tables/`:

    table{1..4}_<group>_per_seed.csv     long format, one row per (seed, exp)
    table{1..4}_<group>_summary.csv      mean / std / n_seeds for each exp

The four groups follow the paper-facing matrix documented in
`docs/experiment.md`:

    table1_main      physics_baseline / wang_dnn_tl / m1_ours
    table2_transfer  m1_ours / a_t1_no_finetune / a_t2_no_pretrain / a_t3_joint
    table3_arch      m1_ours / m2_mlp / m3_cnn1d / m4_transformer
    table4_physics   m1_ours / a_p1_predict_absolute

Figure 3 (data-scale) is still emitted when those experiments are present,
but is no longer part of the paper matrix.

All MAE/RMSE/Std/T95/Tmax/KaggleScore columns carry explicit `_dB` suffixes;
MSE is in `_dB2`.  Summary CSVs additionally provide a pretty
`<metric>_str = "mean ± std"` column that is easy to paste into papers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_ROOT / "_tables"

# Canonical experiment groupings (order matters for table column layout).
# Aligned with the paper-facing matrix in `docs/experiment.md` and the
# notebook layout in `notebooks/results_viz.ipynb`.
MAIN_EXPS      = ["physics_baseline", "wang_dnn_tl", "m1_ours"]
ARCH_EXPS      = ["m1_ours", "m2_mlp", "m3_cnn1d", "m4_transformer"]
TRANSFER_EXPS  = ["m1_ours", "a_t1_no_finetune", "a_t2_no_pretrain", "a_t3_joint"]
PHYSICS_EXPS   = ["m1_ours", "a_p1_predict_absolute"]
DATA_SCALE_EXPS = [
    "ds_pretrain_25", "ds_pretrain_50", "ds_pretrain_100",
    "ds_finetune_25", "ds_finetune_50", "ds_finetune_100",
]

# Numeric metric columns we aggregate (mean + std + pretty string) across seeds.
NUMERIC_COLS = [
    "MAE_dB", "RMSE_dB", "MSE_dB2", "Std_dB", "T95_dB", "Tmax_dB", "KaggleScore_dB",
    "Public_MAE_dB", "Public_Score_dB", "Private_MAE_dB", "Private_Score_dB",
    "aging_MAE_dB", "shb_MAE_dB", "unseen_MAE_dB", "cosmos_MAE_dB",
    "booster_MAE_dB", "preamp_MAE_dB",
    "pretrain_loss", "finetune_loss", "joint_loss",
    "wall_time_sec",
]
# Columns we carry through as-is (first-seen value; these are arch properties).
STATIC_COLS = ["model", "stages", "params", "predict_absolute",
               "cosmos_ratio", "kaggle_ratio"]


# ---------------------------------------------------------------------- #
# Discovery                                                              #
# ---------------------------------------------------------------------- #
def discover_seeds(results_root: Path = RESULTS_ROOT) -> List[int]:
    seeds = []
    if not results_root.exists():
        return seeds
    for p in results_root.iterdir():
        if p.is_dir() and p.name.startswith("seed_"):
            tail = p.name.split("_", 1)[1]
            try:
                seeds.append(int(tail))
            except ValueError:
                continue
    return sorted(seeds)


def _load_metrics(seed: int, exp_name: str, results_root: Path = RESULTS_ROOT) -> Optional[Dict[str, Any]]:
    p = results_root / f"seed_{seed}" / exp_name / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _row_from_metrics(exp_name: str, m: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one metrics.json into a single-row dict (seed field set by caller).

    All MAE/RMSE/Std/T95/Tmax/KaggleScore values are in **dB**; MSE in **dB^2**.
    """
    ev = m.get("evaluation", {}) or {}
    ov = ev.get("overall", {})
    by_u = ev.get("by_usage", {})
    by_c = ev.get("by_category", {})
    by_et = ev.get("by_edfa_type", {})

    def g(d, k):
        return d.get(k, float("nan")) if isinstance(d, dict) else float("nan")

    return {
        "experiment": exp_name,
        "model": m.get("model_name"),
        "stages": ",".join(m.get("stages", []) or []),
        "params": m.get("params"),
        "cosmos_ratio": m.get("cosmos_ratio"),
        "kaggle_ratio": m.get("kaggle_ratio"),
        "predict_absolute": m.get("predict_absolute"),
        "MAE_dB":           g(ov, "MAE"),
        "RMSE_dB":          g(ov, "RMSE"),
        "MSE_dB2":          g(ov, "MSE"),
        "Std_dB":           g(ov, "Std"),
        "T95_dB":           g(ov, "T95"),
        "Tmax_dB":          g(ov, "Tmax"),
        "KaggleScore_dB":   g(ov, "KaggleScore"),
        "Public_MAE_dB":    g(by_u.get("Public", {}), "MAE"),
        "Public_Score_dB":  g(by_u.get("Public", {}), "KaggleScore"),
        "Private_MAE_dB":   g(by_u.get("Private", {}), "MAE"),
        "Private_Score_dB": g(by_u.get("Private", {}), "KaggleScore"),
        "aging_MAE_dB":     g(by_c.get("aging", {}), "MAE"),
        "shb_MAE_dB":       g(by_c.get("shb", {}), "MAE"),
        "unseen_MAE_dB":    g(by_c.get("unseen", {}), "MAE"),
        "cosmos_MAE_dB":    g(by_c.get("cosmos", {}), "MAE"),
        "booster_MAE_dB":   g(by_et.get("booster", {}), "MAE"),
        "preamp_MAE_dB":    g(by_et.get("preamp", {}), "MAE"),
        "pretrain_loss": (m.get("stage_losses") or {}).get("pretrain"),
        "finetune_loss": (m.get("stage_losses") or {}).get("finetune"),
        "joint_loss":    (m.get("stage_losses") or {}).get("joint"),
        "wall_time_sec": m.get("wall_time_sec"),
    }


# ---------------------------------------------------------------------- #
# Per-seed long table + summary across seeds                             #
# ---------------------------------------------------------------------- #
def per_seed_long_table(exps: List[str], seeds: List[int],
                        results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """One row per (experiment, seed) — the raw data consumed by summary."""
    rows = []
    missing: List[Tuple[str, int]] = []
    for exp in exps:
        for seed in seeds:
            m = _load_metrics(seed, exp, results_root=results_root)
            if m is None:
                missing.append((exp, seed))
                continue
            row = _row_from_metrics(exp, m)
            row["seed"] = seed
            rows.append(row)
    if missing:
        grouped: Dict[str, List[int]] = {}
        for exp, seed in missing:
            grouped.setdefault(exp, []).append(seed)
        for exp, ss in grouped.items():
            print(f"[aggregate]   missing metrics.json for {exp} @ seeds {ss}")
    df = pd.DataFrame(rows)
    if not df.empty:
        # Put the identifying columns first.
        front = [c for c in ["experiment", "seed"] + STATIC_COLS if c in df.columns]
        other = [c for c in df.columns if c not in front]
        df = df[front + other]
    return df


def summary_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Group by experiment, compute mean/std/n_seeds for every numeric column."""
    if long_df.empty:
        return pd.DataFrame()
    group = long_df.groupby("experiment", sort=False, dropna=False)
    summary_rows = []
    for exp_name, sub in group:
        out: Dict[str, Any] = {"experiment": exp_name, "n_seeds": int(len(sub))}
        # Preserve static columns (take first row's value).
        for col in STATIC_COLS:
            if col in sub.columns:
                out[col] = sub[col].iloc[0]
        # Include the list of seeds for traceability.
        out["seeds"] = ",".join(str(int(s)) for s in sorted(sub["seed"].dropna().tolist()))
        for col in NUMERIC_COLS:
            if col not in sub.columns:
                continue
            values = sub[col].astype(float).values
            if np.all(np.isnan(values)):
                mean = std = float("nan")
            else:
                mean = float(np.nanmean(values))
                std = float(np.nanstd(values, ddof=0))
            out[f"{col}_mean"] = mean
            out[f"{col}_std"] = std
            out[f"{col}_str"] = (
                f"{mean:.4f} ± {std:.4f}" if not np.isnan(mean) else ""
            )
        summary_rows.append(out)
    df = pd.DataFrame(summary_rows)
    # Order columns: identifying → pretty strings for quick paper lookup → raw numbers.
    front = [c for c in ["experiment", "n_seeds", "seeds"] + STATIC_COLS if c in df.columns]
    pretty = [f"{c}_str" for c in NUMERIC_COLS if f"{c}_str" in df.columns]
    numeric = [f"{c}_{stat}" for c in NUMERIC_COLS for stat in ("mean", "std")
               if f"{c}_{stat}" in df.columns]
    return df[front + pretty + numeric]


# ---------------------------------------------------------------------- #
# Data-scale figure table (long format already; still adds summary)      #
# ---------------------------------------------------------------------- #
def data_scale_long(exps: List[str], seeds: List[int],
                    results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows = []
    for exp in exps:
        for seed in seeds:
            m = _load_metrics(seed, exp, results_root=results_root)
            if m is None:
                continue
            r = _row_from_metrics(exp, m)
            r["seed"] = seed
            if exp.startswith("ds_pretrain_"):
                r["axis"] = "pretrain"
                r["ratio"] = r["cosmos_ratio"]
            elif exp.startswith("ds_finetune_"):
                r["axis"] = "finetune"
                r["ratio"] = r["kaggle_ratio"]
            else:
                r["axis"] = "unknown"
                r["ratio"] = None
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #
def main():
    global RESULTS_ROOT, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(RESULTS_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Restrict aggregation to these seeds (default: auto-discover).")
    args = ap.parse_args()

    RESULTS_ROOT = Path(args.results_root)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds if args.seeds else discover_seeds(RESULTS_ROOT)
    if not seeds:
        print(f"[aggregate] no seed_* subdirs under {RESULTS_ROOT}; nothing to do.")
        return 0
    print(f"[aggregate] using seeds: {seeds}")

    groups: List[Tuple[str, List[str]]] = [
        ("table1_main",     MAIN_EXPS),
        ("table2_transfer", TRANSFER_EXPS),
        ("table3_arch",     ARCH_EXPS),
        ("table4_physics",  PHYSICS_EXPS),
    ]
    for name, exps in groups:
        long_df = per_seed_long_table(exps, seeds, results_root=RESULTS_ROOT)
        summ_df = summary_table(long_df)

        per_seed_path = OUT_DIR / f"{name}_per_seed.csv"
        summary_path = OUT_DIR / f"{name}_summary.csv"
        long_df.to_csv(per_seed_path, index=False)
        summ_df.to_csv(summary_path, index=False)
        print(f"\n[aggregate] {name}: {len(long_df)} per-seed rows -> {per_seed_path.name}")
        print(f"[aggregate] {name}: summary ({len(summ_df)} exps) -> {summary_path.name}  (all dB; MSE in dB^2)")
        if not summ_df.empty:
            show_cols = [c for c in ["experiment", "n_seeds", "seeds",
                                     "MAE_dB_str", "RMSE_dB_str", "KaggleScore_dB_str"]
                         if c in summ_df.columns]
            print(summ_df[show_cols].to_string(index=False))

    # Data scale
    ds_long = data_scale_long(DATA_SCALE_EXPS, seeds, results_root=RESULTS_ROOT)
    ds_summary = summary_table(ds_long) if not ds_long.empty else pd.DataFrame()
    (OUT_DIR / "figure3_data_scale_per_seed.csv").write_text(ds_long.to_csv(index=False))
    (OUT_DIR / "figure3_data_scale_summary.csv").write_text(ds_summary.to_csv(index=False))
    print(f"\n[aggregate] figure3_data_scale: {len(ds_long)} per-seed rows")
    if not ds_summary.empty:
        keep = [c for c in ["experiment", "n_seeds", "seeds",
                            "MAE_dB_str", "KaggleScore_dB_str"]
                if c in ds_summary.columns]
        print(ds_summary[keep].to_string(index=False))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
