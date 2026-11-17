#!/usr/bin/env python3
"""Evaluation driver for the feature-source x budget mechanism analysis.

Runs tools/eval_generalist_at_L.py over the checkpoints produced by
tools/run_feature_source_chain.py:

- static_only_s42 / history_only_s42 at L in {14,30,90,180,365,full};
- generalist_s43 / generalist_s44 at L in {14,90,full} (three-seed table);
- optional lane-C checkpoints (direct BCE / STGCN full history) at L=full.

Resume-safe: existing eval JSONs are skipped. Run after the training chain
finished (evaluations use the GPU, so never concurrently with training).

Usage:
    python tools/run_feature_source_evals.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "runs_results" / "feature_source"
BUDGETS = [14, 30, 90, 180, 365, 0]
SEED_BUDGETS = [14, 90, 0]

JOBS = [
    ("static_only_s42", BUDGETS),
    ("history_only_s42", BUDGETS),
    ("generalist_s43", SEED_BUDGETS),
    ("generalist_s44", SEED_BUDGETS),
    ("direct_bce_s43", [0]),
    ("direct_bce_s44", [0]),
    ("stgcn_s43", [0]),
    ("stgcn_s44", [0]),
]


def main() -> int:
    for name, budgets in JOBS:
        run_dir = FS / name
        pts = sorted(run_dir.glob("*.pt")) if run_dir.exists() else []
        if not pts:
            print(f"{name}: no checkpoint yet, skipped", flush=True)
            continue
        for L in budgets:
            out = run_dir / f"eval_L{L}.json"
            if out.exists():
                continue
            cmd = [
                sys.executable, "tools/eval_generalist_at_L.py",
                "--checkpoint", str(pts[0].relative_to(ROOT)),
                "--lookback", str(L),
                "--split", "test",
                "--out", str(out.relative_to(ROOT)),
            ]
            print(f"{name} L={L}: evaluating", flush=True)
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if proc.returncode != 0 or not out.exists():
                print(proc.stdout[-2000:])
                print(proc.stderr[-2000:])
                return 1
    print("feature-source evaluations complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
