#!/usr/bin/env python3
"""Inspect HybridFNOKANPredictor `spectral_gates` across seeds and stages.

Walks `results/seed_*/<experiment>/{pretrain,model}.pt`, extracts the per-layer
`spectral_gates.<i>` scalar parameters, and prints:

    1. Per-seed table: gate values after pretrain vs after finetune.
    2. How much each gate moved during finetune.
    3. Aggregate across seeds: mean abs gate, fraction of seeds where the
       gate is still effectively dead (|gate| < 1e-3).

Use this to decide whether the SpectralMixingLayer is being switched on at
all by training.  If virtually every gate is near zero in every seed, the
hybrid model is in practice equivalent to the no-spectral ablation, and the
weight-decay-on-gates hypothesis is confirmed.

Usage
-----
    python scripts/inspect_spectral_gates.py
    python scripts/inspect_spectral_gates.py --experiments m1_ours a_a2_wo_fourier_kan
    python scripts/inspect_spectral_gates.py --results-root results
    python scripts/inspect_spectral_gates.py --seeds 42 43 44 45
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
# Default to experiments that have spectral mixing enabled.
DEFAULT_EXPERIMENTS = [
    "m1_ours",
    "a_a2_wo_fourier_kan",
    "a_p1_predict_absolute",
    "a_t1_no_finetune",
    "a_t2_no_pretrain",
    "a_t3_joint",
]
DEAD_THRESHOLD = 1e-3   # |gate| below this is treated as effectively zero


def _gates_from_state_dict(sd: Dict) -> List[float]:
    out = []
    for k, v in sd.items():
        if "spectral_gates" in k:
            try:
                out.append(float(v.detach().cpu().flatten()[0].item()))
            except Exception:
                pass
    return out


def _load_state_dict(p: Path) -> Optional[Dict]:
    if not p.exists():
        return None
    try:
        ckpt = torch.load(p, map_location="cpu")
    except Exception as e:
        print(f"[gates] failed to load {p}: {e}")
        return None
    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict") or ckpt.get("model_state_dict")
        if sd is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
        return sd
    return None


def _discover_seeds(root: Path) -> List[int]:
    seeds = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("seed_"):
            tail = p.name.split("_", 1)[1]
            try:
                seeds.append(int(tail))
            except ValueError:
                continue
    return sorted(seeds)


def _fmt_gates(gates: List[float]) -> str:
    if not gates:
        return "(none)"
    return "[" + ", ".join(f"{g:+.4f}" for g in gates) + "]"


def inspect(results_root: Path, experiments: List[str], seeds: List[int]) -> None:
    if not seeds:
        print(f"[gates] no seed_* directories under {results_root}")
        return

    summary: Dict[str, Dict[str, List[float]]] = {
        # exp -> stage -> list of |gate| values flattened across seeds
        e: {"pretrain": [], "finetune": []} for e in experiments
    }

    for exp in experiments:
        any_found = False
        print("\n" + "=" * 78)
        print(f"experiment: {exp}")
        print("=" * 78)
        print(f"{'seed':>5s}  {'stage':9s}  {'max_abs':>8s}  gates")
        for seed in seeds:
            seed_dir = results_root / f"seed_{seed}" / exp
            pt_path = seed_dir / "pretrain.pt"
            ft_path = seed_dir / "model.pt"
            pt_sd = _load_state_dict(pt_path)
            ft_sd = _load_state_dict(ft_path)

            row_pre, row_ft = None, None
            if pt_sd is not None:
                row_pre = _gates_from_state_dict(pt_sd)
            if ft_sd is not None:
                row_ft = _gates_from_state_dict(ft_sd)

            if row_pre is None and row_ft is None:
                continue
            any_found = True

            if row_pre is not None:
                summary[exp]["pretrain"].extend(abs(g) for g in row_pre)
                m = max((abs(g) for g in row_pre), default=float("nan"))
                print(f"{seed:>5d}  {'pretrain':9s}  {m:>8.4f}  {_fmt_gates(row_pre)}")
            if row_ft is not None:
                summary[exp]["finetune"].extend(abs(g) for g in row_ft)
                m = max((abs(g) for g in row_ft), default=float("nan"))
                print(f"{seed:>5d}  {'finetune':9s}  {m:>8.4f}  {_fmt_gates(row_ft)}")

            if row_pre and row_ft and len(row_pre) == len(row_ft):
                deltas = [b - a for a, b in zip(row_pre, row_ft)]
                m = max((abs(d) for d in deltas), default=float("nan"))
                print(f"{seed:>5d}  {'delta':9s}  {m:>8.4f}  {_fmt_gates(deltas)}")
        if not any_found:
            print(f"  (no checkpoints with spectral_gates found for {exp})")

    print("\n" + "#" * 78)
    print("# Aggregate across seeds (one line per experiment x stage)")
    print("#" * 78)
    print(f"{'experiment':28s} {'stage':9s} {'n_layers':>9s} {'mean|g|':>9s} {'max|g|':>9s} {'%dead':>7s}")
    for exp in experiments:
        for stage in ("pretrain", "finetune"):
            vals = summary[exp][stage]
            if not vals:
                continue
            mean_abs = sum(vals) / len(vals)
            max_abs = max(vals)
            dead_frac = 100.0 * sum(1 for g in vals if g < DEAD_THRESHOLD) / len(vals)
            print(f"{exp:28s} {stage:9s} {len(vals):>9d} {mean_abs:>9.4f} {max_abs:>9.4f} {dead_frac:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    ap.add_argument(
        "--experiments", nargs="*", default=DEFAULT_EXPERIMENTS,
        help="Experiment names to inspect (default: spectral-enabled ones).",
    )
    ap.add_argument(
        "--seeds", type=int, nargs="*", default=None,
        help="Seeds to scan (default: auto-discover seed_* under results_root).",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    seeds = args.seeds if args.seeds else _discover_seeds(results_root)
    print(f"[gates] results_root = {results_root}")
    print(f"[gates] seeds        = {seeds}")
    print(f"[gates] experiments  = {args.experiments}")
    inspect(results_root, args.experiments, seeds)


if __name__ == "__main__":
    main()
