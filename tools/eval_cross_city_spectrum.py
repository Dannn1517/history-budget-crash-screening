#!/usr/bin/env python3
"""Cross-city budget-spectrum evaluation under ONE implementation.

The cross-city primary metric is
unconstrained citywide top-k (same-day scores concatenated across the
training partition before selection), applied identically to both cities;
borough-stratified metrics of differently partitioned cities are not
directly comparable.

For each budget, three scorers are evaluated on the same windows:
  ranker      - zero-training frequency channel at the forecast origin
  generalist  - mixed-budget neural retriever checkpoint
  harsf       - frozen-switch HAR-SF checkpoint

The bundle is rebuilt from the HAR-SF checkpoint's recorded args (which
emit freq_origin/log_n); the generalist must share the feature contract.

Outputs six metrics per scorer per budget (citywide scope by default;
--scope stratified reproduces the within-partition protocol for sanity
checks against run summaries).

``--split`` is honored for both the prebuilt-budget path (the checkpoint's
per-budget ``<split>_loaders_by_L``) and the rebuilt/unseen-budget path, so a
requested ``val`` split never silently evaluates ``test`` loaders. The HR
threshold ratios default to the checkpoint's recorded ``hr_ratio``; pass
``--hr_ratio 0.01,0.02,...`` to override (also used for unseen-budget
reconstruction), which is then serialized into the output JSON.

Usage:
  PYTHONPATH=. python tools/eval_cross_city_spectrum.py \
      --harsf-checkpoint runs_results/chicago_formal/safe_fallback_fixed_switch/st_rghs_best.pt \
      --generalist-checkpoint runs_results/chicago_formal/generalist_6ep_shared_stats/st_event_occurrence_weighted_bce_best.pt \
      --budgets 14,21,30,60,90,120,180,365,0 \
      --out analysis_results/cross_city_spectrum_chicago.json
"""

from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from main_baseline import (
    _to_bool_flag,
    build_bundles,
    build_graph_input,
    finalize_det_eval_state,
    init_det_eval_state,
    is_space_metric_key,
    parse_hr_ratios,
    build_parser as baseline_build_parser,
    resolve_device,
    update_det_eval_state,
)
from tools.eval_rghs_at_L import reconstruct_args
from utils.data.expansion import parse_expansion_groups
from utils.tasks import normalize_hist_source_mode, normalize_target_mode

ROOT = Path(__file__).resolve().parents[1]

# Fields that determine the shared graph-bundle contract (input width, output
# width, and per-step feature semantics). Training-only knobs (lr, epochs,
# seed, rghs_* gate params, etc.) are intentionally excluded.
CONTRACT_FIELDS = (
    "data_dir",
    "horizon",
    "history_len",
    "use_priors",
    "use_hist_freq_prior",
    "use_structural_prior",
    "use_recency_prior",
    "use_static_features",
    "use_event_window",
    "use_weather_calendar",
    "hist_budget_channel",
    "hist_source_mode",
    "target_mode",
    "expansion_features",
    "expansion_dir",
    "static_zero_columns",
)

BOOL_CONTRACT_FIELDS = {
    "use_priors",
    "use_hist_freq_prior",
    "use_structural_prior",
    "use_recency_prior",
    "use_static_features",
    "use_event_window",
    "use_weather_calendar",
    "hist_budget_channel",
}

# Parser defaults for fields that older checkpoints may not record at all.
# ``use_static_features``/``use_event_window``/``use_weather_calendar`` were
# added after the original generalist run (their historical semantics are
# "enabled"), and ``static_zero_columns`` was likewise absent from early
# checkpoints. A missing key therefore resolves to its parser default, not to
# ``False``; a key that is *present* with an explicit conflicting value still
# fails the contract check.
CONTRACT_DEFAULTS = {
    "data_dir": "data",
    "horizon": 3,
    "history_len": 14,
    "use_priors": True,
    "use_hist_freq_prior": 1,
    "use_structural_prior": 1,
    "use_recency_prior": 0,
    "use_static_features": 1,
    "use_event_window": 1,
    "use_weather_calendar": 1,
    "hist_budget_channel": 0,
    "hist_source_mode": "raw_risk",
    "expansion_features": "none",
    "expansion_dir": "",
    "static_zero_columns": "",
}


def _contract_value(args: dict, field: str):
    value = args[field] if field in args else CONTRACT_DEFAULTS.get(field)
    if field == "data_dir":
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else ROOT / path).resolve())
    if field in BOOL_CONTRACT_FIELDS:
        return _to_bool_flag(value)
    if field == "hist_source_mode":
        return normalize_hist_source_mode(value)
    if field == "target_mode":
        return None if value is None else normalize_target_mode(value)
    if field == "expansion_features":
        return None if value is None else parse_expansion_groups(value)
    return value


def check_checkpoint_contract(harsf_args: dict, generalist_args: dict) -> None:
    """Reject incompatible horizon / feature-contract fields before combining.

    Raises a clear ``ValueError`` listing every field that would make the shared
    HAR-SF-derived bundle invalid for the generalist's weights (input width and
    per-step feature semantics), while ignoring training-only arguments.
    """
    mismatches = []
    for field in CONTRACT_FIELDS:
        hs = _contract_value(harsf_args, field)
        gs = _contract_value(generalist_args, field)
        if hs != gs:
            mismatches.append(f"{field}: harsf={hs!r} generalist={gs!r}")
    if mismatches:
        raise ValueError(
            "HAR-SF and generalist checkpoints are incompatible for a shared bundle:\n  "
            + "\n  ".join(mismatches)
        )


def select_loaders(bundles, split: str, budget: int):
    """Return ``[bundle[f"{split}_loaders_by_L"][budget] for bundle in bundles]``.

    Fails with a clear ``KeyError`` when a bundle lacks the requested split or
    budget, so an absent budget is never silently mapped onto another split.
    """
    key = f"{split}_loaders_by_L"
    loaders = []
    for i, bundle in enumerate(bundles):
        by_L = bundle.get(key)
        if not by_L or budget not in by_L:
            available = sorted(by_L) if by_L else []
            raise KeyError(
                f"split={split!r} budget={budget!r} is not available for bundle {i}; "
                f"available budgets for {key}: {available}"
            )
        loaders.append(by_L[budget])
    return loaders


def has_loaders_for(bundles, split: str, budget: int) -> bool:
    """True when every bundle carries the requested split/budget loader map."""
    key = f"{split}_loaders_by_L"
    return all(bool(bundle.get(key) and budget in bundle[key]) for bundle in bundles)


def resolve_hr_ratios(cli_hr_ratio, saved_hr_ratio):
    """Return ``(ratios, override)`` from the CLI override or the checkpoint.

    ``cli_hr_ratio is None`` retains the checkpoint's recorded ratios exactly
    (backward compatible); an explicit CLI value is parsed with
    ``parse_hr_ratios`` and reported as an override.
    """
    if cli_hr_ratio is not None:
        return parse_hr_ratios(cli_hr_ratio), True
    return parse_hr_ratios(saved_hr_ratio), False


def load_models(harsf_ckpt, gen_ckpt, bundle, device):
    from models.probabilistic_baselines import STEventOccurrenceBaseline
    from models.reliability_gated import ReliabilityGatedSafeFallback

    hs = dict(harsf_ckpt["args"])
    common = dict(
        input_dim=bundle["graph_input_dim"], hidden_dim=hs["hidden_dim"], horizon=hs["horizon"],
        kernel_size=hs["kernel_size"], temp_layers=hs["hurdle_temp_layers"],
        spatial_layers=hs["hurdle_spatial_layers"], diffusion_order=hs["diffusion_steps"],
        dropout=hs["dropout"], temporal_backbone=hs["temporal_backbone"],
        spatial_backbone=hs["spatial_backbone"], gat_heads=hs["gat_heads"],
        log_n_ref=hs.get("rghs_log_n_ref", math.log(365.0)),
        init_gate_slope=hs.get("rghs_init_gate_slope", 1.0),
        init_gate_bias=hs.get("rghs_init_gate_bias", 0.0),
        init_fused_scale=hs.get("rghs_init_fused_scale", 1.0),
    )
    harsf = ReliabilityGatedSafeFallback(**common)
    harsf.load_state_dict(harsf_ckpt["model"])

    gs = dict(gen_ckpt["args"])
    gen = STEventOccurrenceBaseline(
        input_dim=bundle["graph_input_dim"], hidden_dim=gs["hidden_dim"], horizon=gs["horizon"],
        kernel_size=gs["kernel_size"], temp_layers=gs["hurdle_temp_layers"],
        spatial_layers=gs["hurdle_spatial_layers"], diffusion_order=gs["diffusion_steps"],
        dropout=gs["dropout"], temporal_backbone=gs["temporal_backbone"],
        spatial_backbone=gs["spatial_backbone"], gat_heads=gs["gat_heads"],
    )
    gen.load_state_dict(gen_ckpt["model"])
    return harsf.to(device=device).eval(), gen.to(device=device).eval()


@torch.no_grad()
def collect_windows(model_h, model_g, bundles, budget, split, ns, device):
    """Per aligned window: scores [3][N_total, H] and labels [N_total, H] (citywide concat)."""
    loaders = select_loaders(bundles, split, budget)
    per_borough = []
    for bundle, loader in zip(bundles, loaders):
        wins = []
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            x = build_graph_input(batch, bundle, use_priors=ns.use_priors)
            logits_h = model_h(x, bundle["A_q"], bundle["A_h"], batch["x_static"],
                               batch["freq_origin"], batch["log_n"]).logits[0].cpu().numpy()
            logits_g = model_g(x, bundle["A_q"], bundle["A_h"])[0].cpu().numpy()
            freq = batch["freq_origin"][0].cpu().numpy()
            labels = (batch["y_event"][0] > 0).float().cpu().numpy()
            wins.append({"harsf": logits_h, "generalist": logits_g,
                         "ranker": np.broadcast_to(freq[:, None], labels.shape).copy(),
                         "labels": labels})
        per_borough.append(wins)
    n_win = len(per_borough[0])
    assert all(len(p) == n_win for p in per_borough), "borough loaders misaligned"
    merged = []
    for w in range(n_win):
        merged.append({
            key: np.concatenate([per_borough[b][w][key] for b in range(len(per_borough))], axis=0)
            for key in ("harsf", "generalist", "ranker", "labels")
        })
    return merged


def eval_scope(windows, scorer, ratios, citywide: bool, borough_sizes=None) -> dict:
    state = init_det_eval_state()
    for win in windows:
        y, s = win["labels"], win[scorer]
        if citywide:
            update_det_eval_state(state, y[None], y[None], np.zeros_like(y)[None], ratios, score=s[None])
        else:
            start = 0
            for n in borough_sizes:
                yb, sb = y[start:start + n], s[start:start + n]
                update_det_eval_state(state, yb[None], yb[None], np.zeros_like(yb)[None], ratios, score=sb[None])
                start += n
    raw = finalize_det_eval_state(state)
    return {f"es_{k}": round(float(v), 4) for k, v in raw.items() if is_space_metric_key(k)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harsf-checkpoint", required=True)
    ap.add_argument("--generalist-checkpoint", required=True)
    ap.add_argument("--budgets", default="14,21,30,60,90,120,180,365,0")
    ap.add_argument("--scope", default="citywide", choices=["citywide", "stratified"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--hr_ratio", default=None, help="comma-separated HR thresholds; default uses the checkpoint's.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]

    harsf_ckpt = torch.load(args.harsf_checkpoint, map_location="cpu", weights_only=False)
    gen_ckpt = torch.load(args.generalist_checkpoint, map_location="cpu", weights_only=False)
    saved_orig = dict(harsf_ckpt["args"])
    saved = dict(harsf_ckpt["args"])
    # Legacy checkpoints predate parser fields that build_bundles now reads.
    # Contract fields resolve to CONTRACT_DEFAULTS (the tool's documented
    # semantics for missing keys); everything else to current parser defaults
    # (feature-inert while expansion_features="none").
    for field, default in CONTRACT_DEFAULTS.items():
        saved.setdefault(field, default)
    for key, value in vars(baseline_build_parser().parse_args([])).items():
        saved.setdefault(key, value)
    ratios, hr_ratio_override = resolve_hr_ratios(args.hr_ratio, saved["hr_ratio"])
    saved["hr_ratio"] = ratios
    saved["device"] = args.device
    saved["progress_interval"] = 0
    check_checkpoint_contract(saved, dict(gen_ckpt["args"]))
    ns = Namespace(**saved)
    device = resolve_device(args.device)
    bundles = build_bundles(ns, device=device)
    model_h, model_g = load_models(harsf_ckpt, gen_ckpt, bundles[0], device)
    borough_sizes = [b["num_nodes"] for b in bundles]

    out = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).date().isoformat(),
        "scope": args.scope,
        "split": args.split,
        "hr_ratio": ratios,
        "hr_ratio_override": hr_ratio_override,
        "data_dir": saved.get("data_dir"),
        "harsf_checkpoint": args.harsf_checkpoint,
        "generalist_checkpoint": args.generalist_checkpoint,
        "borough_sizes": borough_sizes,
        "by_budget": {},
    }
    for budget in budgets:
        if has_loaders_for(bundles, args.split, budget):
            cur_bundles, cur_ns = bundles, ns
        else:
            # unseen budget: rebuild preprocessing exactly as at training time
            # (two-variant mix under shared statistics), same as eval_rghs_at_L
            eval_args = reconstruct_args(saved_orig, budget, device=args.device)
            eval_args.use_priors = _to_bool_flag(eval_args.use_priors)
            eval_args.use_hist_freq_prior = _to_bool_flag(eval_args.use_hist_freq_prior)
            eval_args.use_structural_prior = _to_bool_flag(eval_args.use_structural_prior)
            eval_args.target_mode = normalize_target_mode(eval_args.target_mode)
            eval_args.hist_source_mode = normalize_hist_source_mode(eval_args.hist_source_mode)
            eval_args.hr_ratio = ratios
            cur_bundles = build_bundles(eval_args, device=device)
            cur_ns = eval_args
        windows = collect_windows(model_h, model_g, cur_bundles, budget, args.split, cur_ns, device)
        row = {}
        for scorer in ("ranker", "generalist", "harsf"):
            row[scorer] = eval_scope(windows, scorer, ratios,
                                     citywide=(args.scope == "citywide"),
                                     borough_sizes=borough_sizes)
        row["margin_generalist_minus_ranker_R20"] = round(
            row["generalist"]["es_recall_at_20"] - row["ranker"]["es_recall_at_20"], 4)
        row["margin_harsf_minus_ranker_R20"] = round(
            row["harsf"]["es_recall_at_20"] - row["ranker"]["es_recall_at_20"], 4)
        out["by_budget"][str(budget)] = row
        print(f"L={budget}: ranker {row['ranker']['es_recall_at_20']:.4f} | "
              f"gen {row['generalist']['es_recall_at_20']:.4f} | harsf {row['harsf']['es_recall_at_20']:.4f} | "
              f"margin {row['margin_generalist_minus_ranker_R20']:+.4f}", flush=True)
        del windows
        if cur_bundles is not bundles:
            del cur_bundles
            torch.cuda.empty_cache()

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
