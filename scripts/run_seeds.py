#!/usr/bin/env python3
"""Run the experiment matrix across multiple seeds, sequentially.

Thin wrapper over `scripts/run_matrix.py`.  For each seed it invokes
`run_matrix --seed <S>` with the same pass-through flags (``--only`` /
``--force`` / ``--override`` / ``--no-cache`` / ``--skip-failed`` /
``--include-data-scale`` / ``--dry-run``), so each seed has its own
checkpoint-resume state file under ``results/seed_<S>/_run_state.json``.

Usage
-----
    python scripts/run_seeds.py --seeds 42 43 44 45 46
    python scripts/run_seeds.py --seeds 43 44 45 --only m1_ours a_a1_wo_spectral
    python scripts/run_seeds.py --seeds 43 44 --dry-run
    python scripts/run_seeds.py --seeds 43 44 --aggregate-after

If a seed's run_matrix returns non-zero (failures or user Ctrl+C), the
remaining seeds still execute unless ``--stop-on-failure`` is passed.

Outputs
-------
Per seed:   results/seed_<S>/
Aggregates: results/_tables/               (via aggregate_results.py)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_MATRIX = PROJECT_ROOT / "scripts" / "run_matrix.py"
AGGREGATE = PROJECT_ROOT / "scripts" / "aggregate_results.py"


def _fmt_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:>2d}h{m:02d}m{s:02d}s" if h else f"{m:>3d}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="Seeds to run sequentially, e.g. --seeds 42 43 44 45 46")
    ap.add_argument("--only", nargs="*", default=None, help="Passed through to run_matrix.")
    ap.add_argument("--force", action="store_true", help="Passed through to run_matrix.")
    ap.add_argument("--skip-failed", action="store_true", help="Passed through to run_matrix.")
    ap.add_argument("--no-cache", action="store_true", help="Passed through to run_matrix.")
    ap.add_argument("--include-data-scale", action="store_true", help="Passed through to run_matrix.")
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides, passed through.")
    ap.add_argument("--dry-run", action="store_true", help="Passed through to run_matrix.")
    ap.add_argument("--stop-on-failure", action="store_true",
                    help="Stop iterating further seeds if any run_matrix call returns non-zero.")
    ap.add_argument("--aggregate-after", action="store_true",
                    help="Run scripts/aggregate_results.py after all seeds complete (skipped on --dry-run).")
    args = ap.parse_args()

    per_seed: List[tuple[int, int, float]] = []
    overall_t0 = time.time()
    print(f"[run_seeds] will run {len(args.seeds)} seed(s): {args.seeds}")

    for seed in args.seeds:
        cmd = [sys.executable, str(RUN_MATRIX), "--seed", str(seed)]
        if args.only:
            cmd += ["--only", *args.only]
        if args.force:
            cmd.append("--force")
        if args.skip_failed:
            cmd.append("--skip-failed")
        if args.no_cache:
            cmd.append("--no-cache")
        if args.include_data_scale:
            cmd.append("--include-data-scale")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.override:
            cmd += ["--override", *args.override]

        header = f"SEED {seed}  @ {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        print("\n" + "=" * 80)
        print(header)
        print("  " + " ".join(cmd))
        print("=" * 80, flush=True)

        t0 = time.time()
        try:
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            print("[run_seeds] KeyboardInterrupt - stopping outer loop.")
            dt = time.time() - t0
            per_seed.append((seed, 130, dt))
            break
        dt = time.time() - t0
        per_seed.append((seed, rc, dt))
        if rc != 0:
            print(f"[run_seeds] seed={seed} exited rc={rc} (duration {_fmt_duration(dt)}).")
            if args.stop_on_failure:
                print("[run_seeds] --stop-on-failure set; halting remaining seeds.")
                break
        else:
            print(f"[run_seeds] seed={seed} OK (duration {_fmt_duration(dt)}).")

    overall_dt = time.time() - overall_t0

    # Summary board
    print("\n" + "#" * 80)
    print(f"# run_seeds summary  (total {_fmt_duration(overall_dt)})")
    print("#" * 80)
    print(f"  {'seed':>6s}  {'status':10s}  {'duration':>10s}")
    for seed, rc, dt in per_seed:
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  {seed:>6d}  {status:10s}  {_fmt_duration(dt):>10s}")
    missing = [s for s in args.seeds if s not in {x for x, _, _ in per_seed}]
    for s in missing:
        print(f"  {s:>6d}  {'SKIPPED':10s}  {'':>10s}")
    print("#" * 80)

    all_ok = all(rc == 0 for _, rc, _ in per_seed) and not missing

    if args.aggregate_after and not args.dry_run:
        print("\n[run_seeds] running aggregate_results.py ...")
        subprocess.call([sys.executable, str(AGGREGATE)])

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
