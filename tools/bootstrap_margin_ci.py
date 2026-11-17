#!/usr/bin/env python3
"""Paired moving-block bootstrap CIs for the headline budget-spectrum margins.

Dumps per-window ranking units (hits, denominators, NDCG) for three scorers at
every training budget: the zero-training frequency ranker (the forecast-origin
frequency channel), the mixed-budget generalist, and the frozen-switch HAR-SF.
Pooling the units reproduces the published aggregate metrics; the dump is
validated against the frozen analysis JSONs to four decimals before any
inference is drawn.

Confidence intervals use a cyclic moving-block bootstrap over forecast windows
within each borough (blocks of consecutive days preserve short-range temporal
dependence; windows are the unit of the per-window top-k protocol). Pairs:
generalist-ranker, harsf-ranker, harsf-generalist, per budget, per top-k
ratio, for Recall and NDCG.

Outputs:
  - analysis_results/bootstrap_margin_units.npz  (per-window unit cache)
  - analysis_results/bootstrap_margin_ci.json    (point estimates + 95% CIs)

Usage:
    PYTHONPATH=. python tools/bootstrap_margin_ci.py [--budgets 14,30,90,180,0]
        [--block-len 14] [--n-boot 2000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from main_baseline import (
    build_bundles,
    build_graph_input,
    parse_hr_ratios,
    resolve_device,
)
from main_baseline import build_parser as baseline_build_parser

ROOT = Path(__file__).resolve().parents[1]
SCORERS = ("ranker", "generalist", "harsf")
QUANTITIES = ("hits", "denoms", "ndcg", "valid")
DEFAULT_HARSF = "runs_results/history_adaptive_formal/safe_fallback_fixed_switch_seed42_clean/st_rghs_best.pt"
DEFAULT_GENERALIST = "runs_results/history_adaptive_formal/generalist_6ep_shared_stats/st_event_occurrence_weighted_bce_best.pt"


def load_models(harsf_ckpt, gen_ckpt, bundle, device):
    from models.probabilistic_baselines import STEventOccurrenceBaseline
    from models.reliability_gated import ReliabilityGatedSafeFallback

    hs = dict(harsf_ckpt["args"])
    common = dict(
        input_dim=bundle["graph_input_dim"],
        hidden_dim=hs["hidden_dim"],
        horizon=hs["horizon"],
        kernel_size=hs["kernel_size"],
        temp_layers=hs["hurdle_temp_layers"],
        spatial_layers=hs["hurdle_spatial_layers"],
        diffusion_order=hs["diffusion_steps"],
        dropout=hs["dropout"],
        temporal_backbone=hs["temporal_backbone"],
        spatial_backbone=hs["spatial_backbone"],
        gat_heads=hs["gat_heads"],
        log_n_ref=hs.get("rghs_log_n_ref", math.log(365.0)),
        init_gate_slope=hs.get("rghs_init_gate_slope", 1.0),
        init_gate_bias=hs.get("rghs_init_gate_bias", 0.0),
        init_fused_scale=hs.get("rghs_init_fused_scale", 1.0),
    )
    harsf = ReliabilityGatedSafeFallback(**common)
    harsf.load_state_dict(harsf_ckpt["model"])

    gs = dict(gen_ckpt["args"])
    gen = STEventOccurrenceBaseline(
        input_dim=bundle["graph_input_dim"],
        hidden_dim=gs["hidden_dim"],
        horizon=gs["horizon"],
        kernel_size=gs["kernel_size"],
        temp_layers=gs["hurdle_temp_layers"],
        spatial_layers=gs["hurdle_spatial_layers"],
        diffusion_order=gs["diffusion_steps"],
        dropout=gs["dropout"],
        temporal_backbone=gs["temporal_backbone"],
        spatial_backbone=gs["spatial_backbone"],
        gat_heads=gs["gat_heads"],
    )
    gen.load_state_dict(gen_ckpt["model"])
    return harsf.to(device=device).eval(), gen.to(device=device).eval()


def window_units(score: np.ndarray, labels: np.ndarray, ratios: list[float]) -> dict[str, np.ndarray]:
    """Per-window (hits, denoms, ndcg, valid) shaped [horizon, ratio].

    Mirrors update_det_eval_state exactly: stable descending order per horizon,
    k = max(1, ceil(ratio * N)), accumulation only when the window has a
    positive at that horizon, binary-relevance NDCG with log2 discounting.
    """
    n, horizon = labels.shape
    out = {q: np.zeros((horizon, len(ratios)), dtype=np.float64) for q in QUANTITIES}
    for t in range(horizon):
        y = labels[:, t]
        denoms = float(y.sum())
        order = np.argsort(-score[:, t], kind="mergesort")
        for r_idx, ratio in enumerate(ratios):
            k = max(1, int(math.ceil(float(ratio) * n)))
            top = order[:k]
            if denoms > 0:
                rel = y[top]
                discounts = 1.0 / np.log2(np.arange(2, k + 2))
                dcg = float(np.sum(rel * discounts))
                ideal = min(int(denoms), k)
                idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal + 2))))
                out["hits"][t, r_idx] = float(rel.sum())
                out["denoms"][t, r_idx] = denoms
                out["ndcg"][t, r_idx] = dcg / idcg
                out["valid"][t, r_idx] = 1.0
            else:
                out["denoms"][t, r_idx] = 0.0
    return out


@torch.no_grad()
def dump_budget(model_h, model_g, bundles, budget, ns, device, ratios):
    """Return {scorer: {quantity: array [borough, window, horizon, ratio]}}.

    model_h=None skips the HAR-SF scorer (sync-only sensitivity mode).
    """
    names = SCORERS if model_h is not None else ("ranker", "generalist")
    per_scorer = {s: {q: [] for q in QUANTITIES} for s in names}
    for bundle in bundles:
        loader = bundle["test_loaders_by_L"][budget]
        acc = {s: {q: [] for q in QUANTITIES} for s in names}
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            x = build_graph_input(batch, bundle, use_priors=ns.use_priors)
            logits_g = model_g(x, bundle["A_q"], bundle["A_h"])
            labels = (batch["y_event"] > 0).float()[0].cpu().numpy()
            freq = batch["freq_origin"][0].cpu().numpy()
            score_ranker = np.broadcast_to(freq[:, None], labels.shape)
            scores = {
                "ranker": score_ranker,
                "generalist": logits_g[0].cpu().numpy(),
            }
            if model_h is not None:
                logits_h = model_h(x, bundle["A_q"], bundle["A_h"], batch["x_static"],
                                   batch["freq_origin"], batch["log_n"]).logits
                scores["harsf"] = logits_h[0].cpu().numpy()
            for name, sc in scores.items():
                u = window_units(np.ascontiguousarray(sc), labels, ratios)
                for q in QUANTITIES:
                    acc[name][q].append(u[q])
        for name in names:
            for q in QUANTITIES:
                per_scorer[name][q].append(np.stack(acc[name][q], axis=0))
    return {name: {q: np.stack(per_scorer[name][q], axis=0) for q in QUANTITIES}
            for name in names}


def aggregate(units: dict, scorer: str) -> dict[str, float]:
    """Pooled metrics from per-window units: per-horizon pooling, then horizon mean."""
    hits = units[scorer]["hits"].sum(axis=(0, 1))    # [horizon, ratio]
    denoms = units[scorer]["denoms"].sum(axis=(0, 1))
    ndcg = units[scorer]["ndcg"].sum(axis=(0, 1))
    valid = units[scorer]["valid"].sum(axis=(0, 1))
    out = {}
    recall_t = np.divide(hits, denoms, out=np.full_like(hits, np.nan), where=denoms > 0)
    ndcg_t = np.divide(ndcg, valid, out=np.full_like(ndcg, np.nan), where=valid > 0)
    return out, recall_t, ndcg_t


def cyclic_block_sums(arr: np.ndarray, block_len: int) -> np.ndarray:
    """S[s] = sum of arr[(s+j) % W] for j < block_len; arr is [W, ...]."""
    w = arr.shape[0]
    idx = (np.arange(w)[:, None] + np.arange(block_len)[None, :]) % w
    return arr[idx].sum(axis=1)


def bootstrap_draws(units: dict, borough_windows: list[int], block_len: int,
                    n_boot: int, seed: int, sync_blocks: bool = False) -> dict:
    """Per scorer: bootstrap draws of pooled recall/ndcg, shape [B, ratio].

    sync_blocks=True draws one shared block-origin matrix for all partitions
    (calendar-aligned window axes), preserving cross-partition contemporaneous
    correlation in the pooled draws; the default resamples each partition
    independently.
    """
    rng = np.random.default_rng(seed)
    n_blocks = [int(math.ceil(w / block_len)) for w in borough_windows]
    scorers = tuple(units.keys())
    if sync_blocks:
        if len(set(borough_windows)) != 1 or len(set(n_blocks)) != 1:
            raise ValueError("sync_blocks requires calendar-aligned partition windows")
        shared = rng.integers(0, borough_windows[0], size=(n_boot, n_blocks[0]))
        starts = [shared for _ in borough_windows]
    else:
        starts = [rng.integers(0, w, size=(n_boot, nb)) for w, nb in zip(borough_windows, n_blocks)]
    drawn = {}
    for scorer in scorers:
        # block-sum every quantity per borough, then gather draws
        sums = {q: [] for q in QUANTITIES}
        for b_idx in range(len(borough_windows)):
            blk = {q: cyclic_block_sums(units[scorer][q][b_idx], block_len) for q in QUANTITIES}
            for q in QUANTITIES:
                sums[q].append(blk[q][starts[b_idx]].sum(axis=1))  # [B, horizon, ratio]
        tot = {q: np.sum(sums[q], axis=0) for q in QUANTITIES}
        recall_t = np.divide(tot["hits"], tot["denoms"], out=np.full_like(tot["hits"], np.nan),
                             where=tot["denoms"] > 0)
        ndcg_t = np.divide(tot["ndcg"], tot["valid"], out=np.full_like(tot["ndcg"], np.nan),
                           where=tot["valid"] > 0)
        drawn[scorer] = {
            "recall": np.nanmean(recall_t, axis=1),  # [B, ratio]
            "ndcg": np.nanmean(ndcg_t, axis=1),
        }
    return drawn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harsf-checkpoint", default=DEFAULT_HARSF)
    parser.add_argument("--generalist-checkpoint", default=DEFAULT_GENERALIST)
    parser.add_argument("--budgets", default="14,30,90,180,0")
    parser.add_argument("--block-len", type=int, default=14)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-validate", action="store_true",
                        help="validate against prior_ranker JSONs + run summaries instead of the "
                             "frozen formal artifacts (for reruns whose artifacts are not yet updated)")
    parser.add_argument("--generalist-summary", default="",
                        help="run summary JSON for the generalist checkpoint (used with --no-validate)")
    parser.add_argument("--harsf-summary", default="",
                        help="run summary JSON for the HAR-SF checkpoint (used with --no-validate)")
    parser.add_argument("--out", default="analysis_results/bootstrap_margin_ci.json")
    parser.add_argument("--units-out", default="analysis_results/bootstrap_margin_units.npz")
    parser.add_argument("--sync-only", action="store_true",
                        help="CPU/GPU-light sensitivity mode: rebuild bundles from the "
                             "generalist's own recorded args, score ranker+generalist only, "
                             "and compare independent vs calendar-synchronised block draws "
                             "(synchronised-block sensitivity); writes --sync-out and exits")
    parser.add_argument("--sync-out", default="analysis_results/bootstrap_syncblock_sensitivity.json")
    args = parser.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    harsf_ckpt = torch.load(args.harsf_checkpoint, map_location="cpu", weights_only=False)
    gen_ckpt = torch.load(args.generalist_checkpoint, map_location="cpu", weights_only=False)

    if args.sync_only:
        from models.probabilistic_baselines import STEventOccurrenceBaseline
        # Bundles come from the HAR-SF recorded args (they emit freq_origin for
        # the ranker score and reproduce the published generalist Recall@20%);
        # only the generalist is scored, so HAR-SF model drift is irrelevant.
        saved = dict(harsf_ckpt["args"])
        for key, value in vars(baseline_build_parser().parse_args([])).items():
            saved.setdefault(key, value)
        ratios = parse_hr_ratios(saved["hr_ratio"])
        saved["hr_ratio"] = ratios
        saved["device"] = args.device
        saved["progress_interval"] = 0
        ns = Namespace(**saved)
        device = resolve_device(args.device)
        bundles = build_bundles(ns, device=device)
        gs = dict(gen_ckpt["args"])
        model_g = STEventOccurrenceBaseline(
            input_dim=bundles[0]["graph_input_dim"],
            hidden_dim=gs["hidden_dim"], horizon=gs["horizon"],
            kernel_size=gs["kernel_size"],
            temp_layers=gs["hurdle_temp_layers"],
            spatial_layers=gs["hurdle_spatial_layers"],
            diffusion_order=gs["diffusion_steps"], dropout=gs["dropout"],
            temporal_backbone=gs["temporal_backbone"],
            spatial_backbone=gs["spatial_backbone"], gat_heads=gs["gat_heads"],
        )
        model_g.load_state_dict(gen_ckpt["model"])
        model_g = model_g.to(device=device).eval()
        rows = []
        r_idx = ratios.index(0.2)
        for budget in budgets:
            units = dump_budget(None, model_g, bundles, budget, ns, device, ratios)
            windows = [u.shape[0] for u in units["ranker"]["hits"]]
            ind = bootstrap_draws(units, windows, args.block_len, args.n_boot, args.seed)
            syn = bootstrap_draws(units, windows, args.block_len, args.n_boot, args.seed,
                                  sync_blocks=True)
            d_i = ind["generalist"]["recall"][:, r_idx] - ind["ranker"]["recall"][:, r_idx]
            d_s = syn["generalist"]["recall"][:, r_idx] - syn["ranker"]["recall"][:, r_idx]
            lo, hi = np.percentile(d_i, [2.5, 97.5])
            s_lo, s_hi = np.percentile(d_s, [2.5, 97.5])
            rows.append({
                "L": budget,
                "partition_windows": windows,
                "margin_point": round(float(d_i.mean()), 4),
                "ci_independent_blocks": [round(float(lo), 4), round(float(hi), 4)],
                "ci_synchronised_blocks": [round(float(s_lo), 4), round(float(s_hi), 4)],
                "max_endpoint_shift": round(float(max(abs(s_lo - lo), abs(s_hi - hi))), 4),
            })
            print(f"L={budget}: margin={rows[-1]['margin_point']:+.4f} "
                  f"indep {rows[-1]['ci_independent_blocks']} sync {rows[-1]['ci_synchronised_blocks']} "
                  f"shift={rows[-1]['max_endpoint_shift']:.4f}", flush=True)
        document = {
            "schema_version": 1,
            "question": ("Does resampling partition blocks independently (rather than with "
                         "calendar-synchronised blocks across partitions) materially change "
                         "the pooled stratified margin intervals?"),
            "protocol": {
                "bundles_from": "generalist recorded args (feature contract of its training)",
                "scorers": ["ranker", "generalist"],
                "ratio": 0.2,
                "block_len": args.block_len,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "note": ("Partition window axes are calendar-aligned (identical window counts), "
                         "so synchronised draws use one shared block-origin matrix for all "
                         "partitions, preserving cross-partition contemporaneous correlation."),
            },
            "rows": rows,
        }
        os.makedirs(os.path.dirname(args.sync_out) or ".", exist_ok=True)
        with open(args.sync_out, "w", encoding="utf-8") as fh:
            json.dump(document, fh, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"Saved: {args.sync_out}")
        return

    saved = dict(harsf_ckpt["args"])
    # Old checkpoints predate some parser fields; fill them from current defaults.
    for key, value in vars(baseline_build_parser().parse_args([])).items():
        saved.setdefault(key, value)
    ratios = parse_hr_ratios(saved["hr_ratio"])
    saved["hr_ratio"] = ratios
    saved["device"] = args.device
    saved["progress_interval"] = 0
    ns = Namespace(**saved)
    device = resolve_device(args.device)
    bundles = build_bundles(ns, device=device)
    model_h, model_g = load_models(harsf_ckpt, gen_ckpt, bundles[0], device)

    borough_windows = []
    all_units = {}
    for budget in budgets:
        units = dump_budget(model_h, model_g, bundles, budget, ns, device, ratios)
        all_units[budget] = units
        borough_windows = [u.shape[0] for u in units["ranker"]["hits"]]
        print(f"budget L={budget}: units dumped, borough windows={borough_windows}", flush=True)

    # ---- validation: pooled units must reproduce reference aggregates ----
    ratio_suffix = {0.05: "5", 0.1: "10", 0.2: "20"}
    checks, max_diff = [], 0.0
    if args.no_validate:
        # Reference mode for reruns whose frozen artifacts are not yet updated:
        # ranker bars come from the (data-fix-invariant) prior_ranker test JSONs;
        # model rows come from the run summaries' own test_metrics_by_L.
        refs: dict = {"ranker": {}, "generalist": {}, "harsf": {}}
        for budget in budgets:
            doc = json.loads((ROOT / "analysis_results" / f"prior_ranker_L{budget}.json").read_text())
            refs["ranker"][budget] = doc["results"]["_pooled"]
        for scorer, path in (("generalist", args.generalist_summary), ("harsf", args.harsf_summary)):
            if path:
                by_L = json.loads((ROOT / path).read_text())["test_metrics_by_L"]
                for budget in budgets:
                    refs[scorer][budget] = by_L[str(budget)]
        for budget in budgets:
            for scorer in SCORERS:
                table = refs[scorer].get(budget)
                if not table:
                    continue
                _, recall_t, ndcg_t = aggregate(all_units[budget], scorer)
                for ratio, suffix in ratio_suffix.items():
                    r_idx = ratios.index(ratio)
                    for metric, key in (("recall", f"es_recall_at_{suffix}"), ("ndcg", f"es_ndcg_at_{suffix}")):
                        got = float(np.nanmean((recall_t if metric == "recall" else ndcg_t)[:, r_idx]))
                        diff = abs(got - table[key])
                        max_diff = max(max_diff, diff)
                        checks.append({"L": budget, "scorer": scorer, "metric": key,
                                       "recomputed": round(got, 4), "published": table[key], "diff": round(diff, 6)})
    else:
        fmt = json.loads((ROOT / "analysis_results" / "formal_main_table.json").read_text())
        spectrum = {int(row["L"]): row for row in fmt["headline_spectrum"]}
        harsf_six = json.loads((ROOT / "analysis_results" / "harsf_fixed_formal.json").read_text())["full_six_metrics"]
        gen_six = fmt["shared_stats_full_six"]
        for budget in budgets:
            agg = {}
            for scorer in SCORERS:
                _, recall_t, ndcg_t = aggregate(all_units[budget], scorer)
                agg[scorer] = {"recall": float(np.nanmean(recall_t[:, ratios.index(0.2)])),
                               "ndcg": float(np.nanmean(ndcg_t[:, ratios.index(0.2)])),
                               "recall_t": recall_t, "ndcg_t": ndcg_t}
            row = spectrum[str(budget)] if str(budget) in spectrum else spectrum.get(budget)
            pairs = [("ranker", row["ranker_R20"], "R20"), ("generalist", row["generalist_R20"], "R20"),
                     ("harsf", row["harsf_R20"], "R20")]
            for scorer, want, _ in pairs:
                diff = abs(agg[scorer]["recall"] - want)
                max_diff = max(max_diff, diff)
                checks.append({"L": budget, "scorer": scorer, "metric": "R20",
                               "recomputed": round(agg[scorer]["recall"], 4), "published": want, "diff": round(diff, 6)})
            six = {"harsf": harsf_six.get(str(budget)), "generalist": gen_six.get(str(budget))}
            for scorer, table in six.items():
                if not table:
                    continue
                for ratio, suffix in ratio_suffix.items():
                    r_idx = ratios.index(ratio)
                    for metric, key in (("recall", f"es_recall_at_{suffix}"), ("ndcg", f"es_ndcg_at_{suffix}")):
                        got = float(np.nanmean(agg[scorer][f"{metric}_t"][:, r_idx]))
                        diff = abs(got - table[key])
                        max_diff = max(max_diff, diff)
                        checks.append({"L": budget, "scorer": scorer, "metric": key,
                                       "recomputed": round(got, 4), "published": table[key], "diff": round(diff, 6)})
    validation_ok = max_diff <= 1e-4
    print(f"validation: {len(checks)} aggregate checks, max abs diff {max_diff:.2e} -> {'OK' if validation_ok else 'FAIL'}")
    if not validation_ok:
        bad = [c for c in checks if c["diff"] > 1e-4]
        print(json.dumps(bad[:10], indent=1))
        raise SystemExit("Per-window dump does not reproduce published aggregates; aborting.")

    # ---- ranker six-metric table (feeds the spectrum @5/@10 extension) ----
    ranker_six = {}
    for budget in budgets:
        _, recall_t, ndcg_t = aggregate(all_units[budget], "ranker")
        ranker_six[str(budget)] = {}
        for ratio, suffix in ratio_suffix.items():
            r_idx = ratios.index(ratio)
            ranker_six[str(budget)][f"es_recall_at_{suffix}"] = round(float(np.nanmean(recall_t[:, r_idx])), 4)
            ranker_six[str(budget)][f"es_ndcg_at_{suffix}"] = round(float(np.nanmean(ndcg_t[:, r_idx])), 4)

    # ---- bootstrap ----
    pairs = [("generalist_minus_ranker", "generalist", "ranker"),
             ("harsf_minus_ranker", "harsf", "ranker"),
             ("harsf_minus_generalist", "harsf", "generalist")]
    results = {}
    for budget in budgets:
        drawn = bootstrap_draws(all_units[budget], borough_windows, args.block_len, args.n_boot, args.seed)
        # Calendar-synchronised counterpart: one shared block-origin matrix for
        # all partitions, preserving cross-partition contemporaneous correlation
        # (synchronised-block sensitivity). Same units, same seed family, reported beside
        # the independent-partition intervals.
        sync_drawn = bootstrap_draws(all_units[budget], borough_windows, args.block_len,
                                     args.n_boot, args.seed, sync_blocks=True)
        agg_t = {}
        for scorer in SCORERS:
            _, recall_t, ndcg_t = aggregate(all_units[budget], scorer)
            agg_t[scorer] = {"recall": recall_t, "ndcg": ndcg_t}
        results[str(budget)] = {}
        for ratio, suffix in ratio_suffix.items():
            r_idx = ratios.index(ratio)
            results[str(budget)][f"at_{suffix}"] = {}
            for metric in ("recall", "ndcg"):
                point = {s: float(np.nanmean(agg_t[s][metric][:, r_idx])) for s in SCORERS}
                block = {}
                for name, a, b in pairs:
                    delta_draws = drawn[a][metric][:, r_idx] - drawn[b][metric][:, r_idx]
                    lo, hi = np.percentile(delta_draws, [2.5, 97.5])
                    sync_delta = sync_drawn[a][metric][:, r_idx] - sync_drawn[b][metric][:, r_idx]
                    s_lo, s_hi = np.percentile(sync_delta, [2.5, 97.5])
                    block[name] = {
                        "point_delta": round(point[a] - point[b], 4),
                        "ci95": [round(float(lo), 4), round(float(hi), 4)],
                        "ci95_synchronised_blocks": [round(float(s_lo), 4), round(float(s_hi), 4)],
                        "sync_max_endpoint_shift": round(
                            float(max(abs(s_lo - lo), abs(s_hi - hi))), 4),
                        "boot_se": round(float(delta_draws.std(ddof=1)), 4),
                        "p_le_0": round(float((delta_draws <= 0).mean()), 4),
                    }
                block["point"] = {s: round(point[s], 4) for s in SCORERS}
                results[str(budget)][f"at_{suffix}"][metric] = block
        print(f"budget L={budget}: bootstrap done", flush=True)

    # ---- persist ----
    if args.units_out:
        flat = {}
        for budget in budgets:
            for scorer in SCORERS:
                for q in QUANTITIES:
                    flat[f"L{budget}_{scorer}_{q}"] = all_units[budget][scorer][q].astype(np.float32)
        np.savez_compressed(ROOT / args.units_out, **flat,
                            borough_windows=np.array(borough_windows),
                            budgets=np.array(budgets), ratios=np.array(ratios))
    payload = {
        "schema_version": 1,
        "created_at": "2026-09-10",
        "question": "Do the headline budget-spectrum margins survive paired window-level inference?",
        "protocol": {
            "unit": "forecast window per borough (the top-k protocol unit); horizons pooled per the published aggregation",
            "bootstrap": f"cyclic moving-block over consecutive windows within borough, block length {args.block_len} days, B={args.n_boot}, seed {args.seed}",
            "bootstrap_synchronised_counterpart": (
                "one shared block-origin matrix across calendar-aligned partitions; "
                "reported per margin as ci95_synchronised_blocks"),
            "scorers": {
                "ranker": "forecast-origin frequency channel (zero training), budget-matched",
                "generalist": args.generalist_checkpoint,
                "harsf": args.harsf_checkpoint,
            },
            "validation": {
                "checks": len(checks),
                "max_abs_diff_vs_published": round(max_diff, 6),
                "status": "OK" if validation_ok else "FAIL",
            },
        },
        "ranker_six_metrics_by_budget": ranker_six,
        "results": results,
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {out_path}")

    # headline summary
    for budget in budgets:
        r = results[str(budget)]["at_20"]["recall"]
        print(f"L={budget}: gen-ranker {r['generalist_minus_ranker']['point_delta']:+.4f} "
              f"[{r['generalist_minus_ranker']['ci95'][0]:+.4f},{r['generalist_minus_ranker']['ci95'][1]:+.4f}] | "
              f"harsf-ranker {r['harsf_minus_ranker']['point_delta']:+.4f} "
              f"[{r['harsf_minus_ranker']['ci95'][0]:+.4f},{r['harsf_minus_ranker']['ci95'][1]:+.4f}] | "
              f"harsf-gen {r['harsf_minus_generalist']['point_delta']:+.4f} "
              f"[{r['harsf_minus_generalist']['ci95'][0]:+.4f},{r['harsf_minus_generalist']['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
