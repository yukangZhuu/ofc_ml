#!/usr/bin/env python3
"""Summarize training history + optional metrics.json for debugging experiments.

Usage:
  python scripts/analyze_experiment_logs.py \\
    --history results/history.csv \\
    --metrics results/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=Path, required=True, help="CSV with stage,epoch,train,val,lr,...")
    p.add_argument("--metrics", type=Path, default=None, help="Optional metrics.json with evaluation breakdown")
    args = p.parse_args()

    h = pd.read_csv(args.history)

    print("=== Per-stage (history.csv) ===")
    for stage in h["stage"].unique():
        sub = h[h["stage"] == stage].copy()
        ok = np.isfinite(sub["train"].to_numpy(dtype=float))
        ok &= np.isfinite(sub["val"].to_numpy(dtype=float))
        sub = sub.loc[ok]
        if sub.empty:
            print(f"\n{stage}: (no numeric rows)")
            continue
        print(f"\n{stage}: n_rows={len(sub)}")
        print(f"  train: min={sub['train'].min():.6g}  last={sub['train'].iloc[-1]:.6g}")
        print(f"  val:   min={sub['val'].min():.6g} @ epoch {sub.loc[sub['val'].idxmin(), 'epoch']:.0f}  last={sub['val'].iloc[-1]:.6g}")
        # RMSE in same units as sqrt(MSE); only meaningful if loss is MSE-type
        vmin = sub["val"].min()
        print(f"  val sqrt(min): {float(np.sqrt(max(vmin, 0.0))):.6g}")

    if args.metrics and args.metrics.exists():
        with open(args.metrics, encoding="utf-8") as f:
            m = json.load(f)
        print("\n=== metrics.json snapshot ===")
        print(f"experiment: {m.get('experiment_name')}  model: {m.get('model_name')}")
        sl = m.get("stage_losses") or {}
        if sl:
            print("stage_losses (stored best / last stage loss — see training code): ", sl)
        ev = m.get("evaluation") or {}
        overall = ev.get("overall") or {}
        if overall:
            print(
                f"\nTest overall: MAE_dB={overall.get('MAE_dB')}  "
                f"RMSE_dB={overall.get('RMSE_dB')}  n_samples={overall.get('n_samples')}"
            )
        by_et = ev.get("by_edfa_type") or {}
        if by_et:
            print("\nBy EDFA_type (often explains a 'good val / bad test' story):")
            for k, v in sorted(by_et.items()):
                print(
                    f"  {k:8s}  MAE_dB={v.get('MAE_dB'):8.4f}  "
                    f"n_samples={v.get('n_samples')}  Tmax_dB={v.get('Tmax_dB')}"
                )
        by_cat = ev.get("by_category") or {}
        if by_cat:
            print("\nBy Category:")
            for k, v in sorted(by_cat.items()):
                print(f"  {k:10s} MAE_dB={v.get('MAE_dB'):8.4f}  n_samples={v.get('n_samples')}")
    elif args.metrics:
        print(f"\n(metrics file not found: {args.metrics})")


if __name__ == "__main__":
    main()
