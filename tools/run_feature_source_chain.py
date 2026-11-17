#!/usr/bin/env python3
"""Feature-source x budget mechanism analysis + seed replication chain.

Serial job runner for the mechanism-analysis and seed-replication jobs:

- generalist (mixed-budget) seeds 43/44  -> 3-seed mean+-std at 14/90/full;
- static_only / history_only mixed-budget runs (dimension-preserving
  feature-group ablation, --feature_ablation) -> feature-source x budget curves;
- direct event BCE full-history seeds 43/44 and STGCN full-history seeds 43/44
  (optional seed-strengthening of the full-history suite).

Base configurations are replayed verbatim from the archived run summaries
(args blocks), so every job reproduces the published protocol exactly except
for the seed / ablation override. Jobs with an existing checkpoint are skipped,
making the chain resume-safe. Two lanes (A/B) may run concurrently on the GPU;
each lane is strictly serial inside itself.

Usage:
    python tools/run_feature_source_chain.py --lane A
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "runs_results" / "feature_source"

GEN_SUMMARY = ROOT / "runs_results/history_adaptive_formal/generalist_6ep_shared_stats/st_event_occurrence_summary_09091909.json"
DCE_SUMMARY = ROOT / "runs_results/review_controls/event_bce_rs/st_event_occurrence_summary_09120454.json"
STG_SUMMARY = ROOT / "runs_results/review_controls/stgcn_event_rs/stgcn_summary_09120546.json"

JOBS = {
    # Lanes A and B must NOT run
    # concurrently: each training process pins ~20 GB of CPU RAM during data
    # loading and two simultaneous loaders get OOM-killed (rc=-9, observed
    # 2026-09-16). Run A to completion, then B.
    "A": [
        ("generalist_s43", GEN_SUMMARY, {"seed": 43}),
        ("static_only_s42", GEN_SUMMARY, {"seed": 42, "feature_ablation": "static_only"}),
    ],
    "B": [
        ("generalist_s44", GEN_SUMMARY, {"seed": 44}),
        ("history_only_s42", GEN_SUMMARY, {"seed": 42, "feature_ablation": "history_only"}),
    ],
    # Optional seed-strengthening of the full-history suite (best effort);
    # run only when lanes A and B are complete and wall time
    # permits. Same RAM constraint: never concurrent with another lane.
    "C": [
        ("direct_bce_s43", DCE_SUMMARY, {"seed": 43}),
        ("direct_bce_s44", DCE_SUMMARY, {"seed": 44}),
        ("stgcn_s43", STG_SUMMARY, {"seed": 43}),
        ("stgcn_s44", STG_SUMMARY, {"seed": 44}),
    ],
}


def build_cli(base_args: dict, overrides: dict) -> list[str]:
    sys.path.insert(0, str(ROOT))
    from main_baseline import build_parser

    parser = build_parser()
    defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}
    option_of = {
        a.dest: (a.option_strings[0] if a.option_strings else f"--{a.dest}")
        for a in parser._actions
    }
    merged = dict(base_args)
    merged.update(overrides)
    cmd = [sys.executable, "main_baseline.py"]
    for key, value in merged.items():
        if key not in defaults:
            print(f"  [warn] arg {key!r} not in parser; skipped", flush=True)
            continue
        if value == defaults[key]:
            continue
        flag = option_of[key]
        if isinstance(value, bool):
            cmd += [flag, "1" if value else "0"]
        elif isinstance(value, (list, tuple)):
            # parser expects comma-separated strings for list-valued args
            cmd += [flag, ",".join(str(v) for v in value)]
        else:
            cmd += [flag, str(value)]
    return cmd


def wait_for_memory(min_available_gb: float = 18.0, timeout_s: int = 900) -> None:
    """Block until the previous job's process has released its CPU RAM.

    Two consecutive jobs overlap otherwise: the finished process still holds
    ~20 GB while the next one loads its borough bundles, and the 30 GB host
    OOM-kills the new loader (rc=-9, observed twice on 2026-09-16).
    """
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                avail_gb = int(line.split()[1]) / 1024.0 / 1024.0
                break
        else:
            return
        if avail_gb >= min_available_gb:
            return
        time.sleep(10)
    print("  [warn] memory wait timed out; continuing anyway", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=sorted(JOBS), required=True)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    first = True
    for name, summary_path, overrides in JOBS[args.lane]:
        if not first:
            wait_for_memory()
        first = False
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if list(out_dir.glob("*.pt")):
            print(f"[lane {args.lane}] {name}: checkpoint exists, skipped", flush=True)
            continue
        base = json.loads(summary_path.read_text())["args"]
        cmd = build_cli(base, {**overrides, "out_dir": str(out_dir.relative_to(ROOT))})
        log_path = out_dir / "run.log"
        print(f"[lane {args.lane}] {name}: starting ({' '.join(cmd[-6:])} ...)", flush=True)
        with log_path.open("w") as log:
            log.write("command: " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not list(out_dir.glob("*.pt")):
            print(f"[lane {args.lane}] {name}: FAILED (rc={proc.returncode}); stopping lane",
                  flush=True)
            return 1
        print(f"[lane {args.lane}] {name}: done", flush=True)
    print(f"[lane {args.lane}] all jobs complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
