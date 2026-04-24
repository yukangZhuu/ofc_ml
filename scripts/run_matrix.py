#!/usr/bin/env python3
"""Sequential matrix runner with state tracking and crash-safe resume.

Orders the experiment YAMLs from light to heavy, delegates each one to
`scripts/run_experiment.py`, keeps a rolling state file at
`results/_run_state.json`, and prints a live progress board on every
transition.

Checkpoint / resume semantics
-----------------------------
State file records per experiment one of:
    pending | running | done | failed | interrupted

- `done`      : `results/<exp>/metrics.json` exists (either from a previous
                run of this script or from `run_experiment.py` directly);
                skipped on re-run unless `--force`.
- `running`   : the script started it but did not observe completion (e.g.
                process killed, machine crashed).  Treated as pending and
                re-launched from scratch.
- `failed`    : non-zero exit from `run_experiment.py`.  Re-launched on
                next invocation; pass `--skip-failed` to leave them alone.
- `interrupted`: Ctrl+C arrived while this experiment was running.  Treated
                 as pending.

Usage
-----
    python scripts/run_matrix.py                       # run all in order
    python scripts/run_matrix.py --force               # rerun everything
    python scripts/run_matrix.py --only m1_ours m2_mlp # restrict to named experiments
    python scripts/run_matrix.py --dry-run             # print plan and exit
    python scripts/run_matrix.py --override pretrain.epochs=40 finetune.epochs=80
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_ROOT = PROJECT_ROOT / "results"
STATE_FILE = RESULTS_ROOT / "_run_state.json"
LIVE_LOG_FILE = RESULTS_ROOT / "_run_matrix.log"
RUN_SINGLE = PROJECT_ROOT / "scripts" / "run_experiment.py"


# ------------------------------------------------------------------ #
# Experiment ordering                                                #
# ------------------------------------------------------------------ #
# Chosen to schedule light / cache-reusing cells first and the heavy
# Transformer run last.  `data_scale` experiments are intentionally
# excluded per project policy; point `--include-data-scale` at this
# script to flip that bit.
ORDERED_EXPERIMENTS: List[str] = [
    # 1. Cheapest: reuses m1 pretrain cache (or runs it itself).
    "experiments/ablation/transfer/a_t1_no_finetune.yaml",
    # 2. Hybrid variant, similar cost to m1.
    "experiments/ablation/arch/a_a1_wo_spectral.yaml",
    # 3. Hybrid with MLP blocks — slightly faster per step.
    "experiments/ablation/arch/a_a2_wo_fourier_kan.yaml",
    # 4. Reference main result; populates pretrain cache reused above.
    "experiments/main/m1_ours.yaml",
    # 5. Same arch as m1, different target parameterization.
    "experiments/ablation/physics/a_p1_predict_absolute.yaml",
    # 6. Same-size MLP baseline.
    "experiments/main/m2_mlp.yaml",
    # 7. Kaggle-only, fast per epoch but many epochs.
    "experiments/ablation/transfer/a_t2_no_pretrain.yaml",
    # 8. Joint pretrain+finetune merge.
    "experiments/ablation/transfer/a_t3_joint.yaml",
    # 9. Medium-weight 1D CNN baseline.
    "experiments/main/m3_cnn1d.yaml",
    # 10. Heaviest — last.
    "experiments/main/m4_transformer.yaml",
]

DATA_SCALE_EXPERIMENTS: List[str] = [
    "experiments/ablation/data_scale/pretrain_25.yaml",
    "experiments/ablation/data_scale/pretrain_50.yaml",
    "experiments/ablation/data_scale/pretrain_100.yaml",
    "experiments/ablation/data_scale/finetune_25.yaml",
    "experiments/ablation/data_scale/finetune_50.yaml",
    "experiments/ablation/data_scale/finetune_100.yaml",
]


# ------------------------------------------------------------------ #
# Small utilities                                                    #
# ------------------------------------------------------------------ #
@dataclass
class RunRecord:
    name: str
    config_path: str
    status: str = "pending"              # pending|running|done|failed|interrupted
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_sec: Optional[float] = None
    return_code: Optional[int] = None
    overall_mae_dB: Optional[float] = None
    overall_kaggle_dB: Optional[float] = None
    error: Optional[str] = None
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunState:
    records: Dict[str, RunRecord] = field(default_factory=dict)
    last_updated: Optional[str] = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": _utcnow_iso(),
            "records": {k: v.to_dict() for k, v in self.records.items()},
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "RunState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except Exception:
            return cls()
        recs = {k: RunRecord(**v) for k, v in (data.get("records") or {}).items()}
        return cls(records=recs, last_updated=data.get("last_updated"))


def _exp_name(yaml_path: Path) -> str:
    return yaml_path.stem


# ------------------------------------------------------------------ #
# Progress board                                                     #
# ------------------------------------------------------------------ #
STATUS_ICON = {
    "pending":     "·",
    "running":     ">",
    "done":        "v",
    "failed":      "x",
    "interrupted": "!",
}


def _fmt_number(x: Optional[float], width: int = 8) -> str:
    if x is None:
        return " " * width
    return f"{x:>{width}.4f}"


def _fmt_duration(sec: Optional[float]) -> str:
    if sec is None:
        return "     -  "
    m, s = divmod(int(sec), 60)
    return f"{m:>3d}m{s:02d}s"


def print_board(state: RunState, order: List[str], current: Optional[str] = None, note: str = "") -> None:
    print("\n" + "=" * 78)
    print(f"Matrix progress @ {datetime.now().isoformat(timespec='seconds')}  {note}")
    print("-" * 78)
    print(f"{'':2s} {'status':10s} {'experiment':28s} {'MAE(dB)':>8s}  {'Score(dB)':>9s}  {'dur':>9s}")
    for n in order:
        r = state.records.get(n)
        if r is None:
            icon = "?"
            status = "missing"
            dur = mae = score = None
        else:
            icon = STATUS_ICON.get(r.status, "?")
            status = r.status
            dur = r.duration_sec
            mae = r.overall_mae_dB
            score = r.overall_kaggle_dB
        marker = "*" if current == n else " "
        print(f"{marker} {icon:1s} {status:8s} {n:28s} {_fmt_number(mae)}  {_fmt_number(score, 9)}  {_fmt_duration(dur)}")
    print("=" * 78, flush=True)


# ------------------------------------------------------------------ #
# Launching a single experiment                                      #
# ------------------------------------------------------------------ #
def _collect_overall_from_metrics(exp_name: str) -> tuple[Optional[float], Optional[float]]:
    p = RESULTS_ROOT / exp_name / "metrics.json"
    if not p.exists():
        return None, None
    try:
        m = json.loads(p.read_text())
        ov = (m.get("evaluation") or {}).get("overall", {})
        return ov.get("MAE"), ov.get("KaggleScore")
    except Exception:
        return None, None


def launch_experiment(rec: RunRecord, overrides: List[str], no_cache: bool, force: bool) -> int:
    """Start run_experiment.py for `rec`, stream its output, return rc."""
    cmd = [
        sys.executable,
        str(RUN_SINGLE),
        "--config",
        str(PROJECT_ROOT / rec.config_path),
    ]
    if force:
        cmd.append("--force")
    if no_cache:
        cmd.append("--no-cache")
    if overrides:
        cmd += ["--override", *overrides]

    exp_log_dir = RESULTS_ROOT / rec.name / "logs"
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    exp_log_file = exp_log_dir / "run_matrix.log"

    print(f"\n>>> [{rec.name}] {shlex.join(cmd)}\n    -> tee {exp_log_file}")
    # Stream + tee to file
    rec.attempts += 1
    rec.status = "running"
    rec.started_at = _utcnow_iso()

    t0 = time.time()
    with exp_log_file.open("w") as flog:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                flog.write(line)
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            raise
        finally:
            rc = proc.wait()
    rec.finished_at = _utcnow_iso()
    rec.duration_sec = float(time.time() - t0)
    rec.return_code = rc
    return rc


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-run experiments even if they already completed.")
    ap.add_argument("--skip-failed", action="store_true", help="Leave previously failed experiments as-is.")
    ap.add_argument("--no-cache", action="store_true", help="Disable pretrain cache reuse in orchestrate()")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to these experiment names (by YAML stem, e.g. m1_ours).")
    ap.add_argument("--include-data-scale", action="store_true",
                    help="Also run the 6 data-scale ablations (default: excluded).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides applied to every experiment.")
    args = ap.parse_args()

    ordered_paths: List[Path] = [PROJECT_ROOT / p for p in ORDERED_EXPERIMENTS]
    if args.include_data_scale:
        ordered_paths += [PROJECT_ROOT / p for p in DATA_SCALE_EXPERIMENTS]

    ordered_paths = [p for p in ordered_paths if p.exists()]
    if args.only:
        wanted = set(args.only)
        ordered_paths = [p for p in ordered_paths if _exp_name(p) in wanted]
        missing = wanted - {_exp_name(p) for p in ordered_paths}
        if missing:
            print(f"[run_matrix] WARNING: --only names not found among orderings: {sorted(missing)}")

    if not ordered_paths:
        print("[run_matrix] no experiments selected.")
        return 1

    state = RunState.load(STATE_FILE)
    order_names: List[str] = []
    # Reconcile state with the ordered plan.
    for p in ordered_paths:
        name = _exp_name(p)
        order_names.append(name)
        rec = state.records.get(name)
        if rec is None:
            rec = RunRecord(name=name, config_path=str(p.relative_to(PROJECT_ROOT)))
            state.records[name] = rec
        # If `metrics.json` exists but state says otherwise, promote to done.
        metrics_path = RESULTS_ROOT / name / "metrics.json"
        if metrics_path.exists() and rec.status != "failed":
            mae, score = _collect_overall_from_metrics(name)
            rec.status = "done"
            rec.overall_mae_dB = mae
            rec.overall_kaggle_dB = score
        # A stale running from a previous crash -> treat as pending.
        if rec.status == "running":
            rec.status = "interrupted"

    state.save(STATE_FILE)

    if args.dry_run:
        print_board(state, order_names, note="(dry-run)")
        return 0

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with LIVE_LOG_FILE.open("a") as flog:
            flog.write(f"\n===== run_matrix START @ {datetime.now().isoformat(timespec='seconds')} =====\n")
            flog.write(f"args = {vars(args)}\n")
    except Exception:
        pass

    print_board(state, order_names, note="(initial)")

    interrupted = False
    for p in ordered_paths:
        name = _exp_name(p)
        rec = state.records[name]

        if rec.status == "done" and not args.force:
            continue
        if rec.status == "failed" and args.skip_failed:
            continue

        try:
            rc = launch_experiment(rec, args.override, no_cache=args.no_cache, force=args.force)
        except KeyboardInterrupt:
            interrupted = True
            rec.status = "interrupted"
            rec.error = "KeyboardInterrupt from run_matrix"
            state.save(STATE_FILE)
            print_board(state, order_names, current=name, note="(KeyboardInterrupt)")
            break

        if rc == 0:
            mae, score = _collect_overall_from_metrics(name)
            rec.overall_mae_dB = mae
            rec.overall_kaggle_dB = score
            rec.status = "done"
            rec.error = None
        else:
            rec.status = "failed"
            rec.error = f"run_experiment.py returned {rc}"

        state.save(STATE_FILE)
        print_board(state, order_names, current=name, note=f"(after {name})")

    # Final summary
    done = sum(1 for n in order_names if state.records[n].status == "done")
    failed = sum(1 for n in order_names if state.records[n].status == "failed")
    interrupted_n = sum(1 for n in order_names if state.records[n].status == "interrupted")
    pending = sum(1 for n in order_names if state.records[n].status == "pending")

    print("\n" + "#" * 78)
    print(f"# Summary: done={done}  failed={failed}  interrupted={interrupted_n}  pending={pending}")
    print(f"# State file : {STATE_FILE}")
    print(f"# Live log   : {LIVE_LOG_FILE}")
    print("#" * 78)
    return 0 if (failed == 0 and not interrupted) else 2


if __name__ == "__main__":
    raise SystemExit(main())
