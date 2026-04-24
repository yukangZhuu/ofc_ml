#!/usr/bin/env python3
"""Run every experiment YAML under `experiments/`, skipping ones already done.

Usage
-----
    python scripts/run_all.py                         # run all 15 experiments
    python scripts/run_all.py --only main             # run only experiments/main/*.yaml
    python scripts/run_all.py --only ablation/arch    # filter by sub-path
    python scripts/run_all.py --force                 # re-run even if metrics.json exists
    python scripts/run_all.py --dry-run               # print the list without running
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUN_SINGLE = PROJECT_ROOT / "scripts" / "run_experiment.py"


def collect_experiments(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*.yaml")):
        if p.name == "base.yaml":
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="Filter experiments whose path contains this substring")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], help="Global key=value overrides")
    args = ap.parse_args()

    exps = collect_experiments(EXPERIMENTS_DIR)
    if args.only:
        exps = [p for p in exps if args.only in str(p.relative_to(PROJECT_ROOT))]

    if not exps:
        print("[run_all] no experiments matched.")
        return 1

    print(f"[run_all] {len(exps)} experiment(s):")
    for p in exps:
        print(f"  - {p.relative_to(PROJECT_ROOT)}")
    if args.dry_run:
        return 0

    n_ok = 0
    n_fail = 0
    for p in exps:
        cmd = [sys.executable, str(RUN_SINGLE), "--config", str(p)]
        if args.force:
            cmd.append("--force")
        if args.override:
            cmd += ["--override", *args.override]
        print(f"\n[run_all] >>> {p.relative_to(PROJECT_ROOT)}")
        rc = subprocess.call(cmd)
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
            print(f"[run_all] FAILED (rc={rc}): {p}")

    print(f"\n[run_all] done. ok={n_ok} failed={n_fail}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
