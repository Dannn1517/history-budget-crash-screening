#!/usr/bin/env python3
"""Evaluate a trained generalist checkpoint at arbitrary history budgets.

Supports budgets never seen during mixed-budget training (e.g., 21/60/120/365)
by rebuilding bundles with a single truncated frequency channel and evaluating
the test split with the exact same borough-stratified per-window Top-k protocol
used during training.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from main_baseline import (
    build_bundles,
    evaluate_event_occurrence_model,
    get_graph_input_dim,
    parse_hr_ratios,
    resolve_device,
)
from models.probabilistic_baselines import STEventOccurrenceBaseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--lookback", type=int, required=True, help="评估用历史预算；0 表示全历史。")
    parser.add_argument("--mask_borough", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"])
    # Older checkpoints predate later argparse fields (e.g. mobility_lag_days);
    # fill gaps from current parser defaults so frozen artifacts stay loadable.
    from main_baseline import build_parser as baseline_build_parser
    for key, value in vars(baseline_build_parser().parse_args([])).items():
        if not hasattr(saved_args, key):
            setattr(saved_args, key, value)
    shared_stats = bool(getattr(saved_args, "hist_prior_shared_stats", 0))
    if shared_stats:
        # Replicate training-time standardization: the lookback-L channel was
        # standardized with full-history train stats, so evaluate via a
        # two-variant mix (L first, full history as the shared-stats reference)
        # whose default loaders pin to the first variant.
        saved_args.hist_prior_lookback_mix = "0" if args.lookback == 0 else f"{args.lookback},0"
        saved_args.hist_prior_lookback = 0
        saved_args.hist_prior_shared_stats = 1
    else:
        saved_args.hist_prior_lookback = int(args.lookback)
        saved_args.hist_prior_lookback_mix = ""
    saved_args.hist_budget_channel = 0
    if args.mask_borough:
        saved_args.mask_borough = args.mask_borough
    saved_args.hr_ratio = parse_hr_ratios(saved_args.hr_ratio)
    saved_args.device = args.device
    saved_args.progress_interval = 0

    device = resolve_device(saved_args.device)
    bundles = build_bundles(saved_args, device=device)
    model = STEventOccurrenceBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=saved_args.use_priors),
        hidden_dim=saved_args.hidden_dim,
        horizon=saved_args.horizon,
        kernel_size=saved_args.kernel_size,
        temp_layers=saved_args.hurdle_temp_layers,
        spatial_layers=saved_args.hurdle_spatial_layers,
        diffusion_order=saved_args.diffusion_steps,
        dropout=saved_args.dropout,
        temporal_backbone=saved_args.temporal_backbone,
        spatial_backbone=saved_args.spatial_backbone,
        gat_heads=saved_args.gat_heads,
    ).to(device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = evaluate_event_occurrence_model(
        model,
        bundles,
        split=args.split,
        device=device,
        hr_ratio=saved_args.hr_ratio,
        use_priors=saved_args.use_priors,
        eval_horizon=saved_args.event_eval_horizon,
    )
    doc = {
        "checkpoint": args.checkpoint,
        "selected_epoch": ckpt.get("selected_epoch"),
        "lookback": args.lookback,
        "seen_in_training": bool(args.lookback in (14, 30, 90, 180, 0)),
        "mask_borough": saved_args.mask_borough,
        "split": args.split,
        "test_metrics": {k: float(v) for k, v in metrics.items()},
    }
    print(json.dumps(doc, indent=2, ensure_ascii=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
