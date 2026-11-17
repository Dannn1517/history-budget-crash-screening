from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.data.dataset import GraphBaselineDataset, HABaselineDataset, XGBBaselineDataset
from utils.data.graph_ops import (
    apply_mask_to_edge_index,
    build_diffusion_supports,
    compute_structural_node_features,
    make_undirected_edge_index,
)
from utils.data.preprocessing import standardize_freq_variants, standardize_node_time_feature
from utils.data.priors import (
    build_days_since_last_event,
    build_exclusive_cumulative_mean,
    build_per_road_trailing_mean,
    build_trailing_exclusive_mean,
    diffuse_risk_time_space,
    parse_lookback_mix,
)
from utils.tasks import normalize_hist_source_mode, normalize_target_mode, uses_diffused_target


def subgraph_label(name: str) -> str:
    return name.replace(" ", "_")


def build_baseline_bundle(
        borough_name: str,
        mask: np.ndarray,
        X_static_full: np.ndarray,
        X_dynamic_sub: np.ndarray,
        dynamic_feature_names: List[str],
        dynamic_feature_stats: dict,
        Y_hist_full: np.ndarray,
        Y_risk_full: np.ndarray,
        Y_crash_full: np.ndarray,
        calendar: np.ndarray,
        weather: np.ndarray,
        borough_id_arr_full: np.ndarray,
        edge_index_full: np.ndarray,
        args,
        train_s: List[int],
        val_s: List[int],
        test_s: List[int],
        train_end: int,
        device: torch.device,
        hetero_lookbacks_full: np.ndarray = None,
        holdout_mask_full: np.ndarray = None,
) -> dict:
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"Borough {borough_name} has no nodes after masking.")
    hetero_lookbacks_sub = (
        np.asarray(hetero_lookbacks_full)[idx] if hetero_lookbacks_full is not None else None
    )
    holdout_sub = (
        np.asarray(holdout_mask_full)[idx].astype(bool) if holdout_mask_full is not None else None
    )
    fit_node_mask = None
    if holdout_sub is not None and bool(holdout_sub.any()):
        fit_node_mask = ~holdout_sub
        if not bool(fit_node_mask.any()):
            raise ValueError(f"Borough {borough_name} has no seen nodes after node holdout.")

    X_static_public_sub = X_static_full[idx].copy().astype(np.float32)
    if not getattr(args, "use_static_features", 1):
        X_static_public_sub = np.zeros_like(X_static_public_sub)
    Y_hist_sub = Y_hist_full[idx].copy()
    Y_risk_sub = Y_risk_full[idx].copy()
    Y_crash_sub = Y_crash_full[idx].copy()
    borough_id_sub = borough_id_arr_full[idx].astype(np.int64, copy=True)
    target_mode = normalize_target_mode(args.target_mode)
    hist_source_mode = normalize_hist_source_mode(args.hist_source_mode)

    edge_index_sub = apply_mask_to_edge_index(edge_index_full, mask)
    edge_index_undirected_sub = make_undirected_edge_index(edge_index_sub)

    # 结构先验特征
    structural_dim = 2
    if args.use_structural_prior:
        X_structural_sub = compute_structural_node_features(edge_index_undirected_sub, idx.size)
    else:
        X_structural_sub = np.zeros((idx.size, structural_dim), dtype=np.float32)

    weather_by_node_sub = weather[borough_id_sub].astype(np.float32)
    if not getattr(args, "use_weather_calendar", 1):
        weather_by_node_sub = np.zeros_like(weather_by_node_sub)
        calendar = np.zeros_like(calendar)

    # 历史频次先验特征
    hist_prior_lookback = int(getattr(args, "hist_prior_lookback", 0))
    lookback_mix = parse_lookback_mix(getattr(args, "hist_prior_lookback_mix", ""))
    X_hist_freq_variants = None
    if args.use_hist_freq_prior:
        if hetero_lookbacks_sub is not None:
            X_hist_freq_sub = build_per_road_trailing_mean(Y_crash_sub, hetero_lookbacks_sub)
            X_hist_freq_sub, hist_mean, hist_std = standardize_node_time_feature(
                X_hist_freq_sub, train_end, fit_node_mask=fit_node_mask
            )
        elif lookback_mix:
            raw_variants = [
                build_exclusive_cumulative_mean(Y_crash_sub)
                if L == 0
                else build_trailing_exclusive_mean(Y_crash_sub, L)
                for L in lookback_mix
            ]
            shared = bool(getattr(args, "hist_prior_shared_stats", 0))
            ref_index = lookback_mix.index(0) if 0 in lookback_mix else int(np.argmax(lookback_mix))
            variant_list, raw_stats = standardize_freq_variants(
                raw_variants,
                train_end,
                shared_stats=shared,
                ref_index=ref_index,
                fit_node_mask=fit_node_mask,
            )
            variant_stats = [dict(s, lookback=L) for s, L in zip(raw_stats, lookback_mix)]
            X_hist_freq_variants = np.stack(variant_list).astype(np.float32)
            X_hist_freq_sub = X_hist_freq_variants[0]
            hist_mean = variant_stats[0]["mean"]
            hist_std = variant_stats[0]["std"]
        else:
            if hist_prior_lookback > 0:
                X_hist_freq_sub = build_trailing_exclusive_mean(Y_crash_sub, hist_prior_lookback)
            else:
                X_hist_freq_sub = build_exclusive_cumulative_mean(Y_crash_sub)
            X_hist_freq_sub, hist_mean, hist_std = standardize_node_time_feature(
                X_hist_freq_sub, train_end, fit_node_mask=fit_node_mask
            )
    else:
        X_hist_freq_sub = np.zeros_like(Y_crash_sub, dtype=np.float32)
        hist_mean = 0.0
        hist_std = 1.0

    # 事故间隔（recency）先验：距上次事故天数（对 14 天窗口之外的长间隔提供新信息）
    X_recency_sub = None
    if getattr(args, "use_recency_prior", 0):
        X_recency_sub = build_days_since_last_event(Y_crash_sub)
        X_recency_sub, _, _ = standardize_node_time_feature(
            X_recency_sub, train_end, fit_node_mask=fit_node_mask
        )

    # 历史预算条件通道：log(L)/log(T) 的显式预算值（混合预算训练时让模型知道当前预算）
    hist_budget_channel = bool(getattr(args, "hist_budget_channel", 0))
    budget_channel_values = None
    if lookback_mix and hist_budget_channel:
        total_steps_sub = Y_crash_sub.shape[1]
        budget_channel_values = [
            float(np.log(L if L > 0 else total_steps_sub) / np.log(total_steps_sub))
            for L in lookback_mix
        ]

    # 风险平滑 target
    if uses_diffused_target(target_mode):
        Y_target_sub = diffuse_risk_time_space(
            Y_risk_sub,
            edge_index_sub,
            alpha=args.diffuse_alpha,
            k=args.diffuse_k,
            hop_weight=args.diffuse_weight,
            undirected=True,
            normalize=args.diffuse_norm,
            zero_thresh=args.risk_diffuse_zero_thresh,
        ).astype(np.float32)
    else:
        Y_target_sub = Y_risk_sub.copy()

    Y_hist_raw_sub = Y_hist_sub.copy()
    Y_hist_target_sub = Y_target_sub.copy()
    if not getattr(args, "use_event_window", 1):
        Y_hist_raw_sub = np.zeros_like(Y_hist_raw_sub)
        Y_hist_target_sub = np.zeros_like(Y_hist_target_sub)
    graph_hist_source_mode = hist_source_mode
    if hist_source_mode == "both" and not uses_diffused_target(target_mode):
        graph_hist_source_mode = "raw_risk"
    ha_hist_source_mode = "target" if hist_source_mode == "both" else hist_source_mode

    Y_rank_sub = Y_target_sub.copy()
    Y_event_sub = Y_risk_sub.copy()
    event_supervision_space = str(getattr(args, "event_supervision_space", "raw")).strip().lower()
    if event_supervision_space == "target":
        Y_sup_sub = (Y_target_sub > 0).astype(np.float32)
    else:
        Y_sup_sub = Y_event_sub
    train_label_idx = (
        np.asarray(train_s, dtype=np.int64)[:, None]
        + int(args.history_len)
        + np.arange(int(args.horizon), dtype=np.int64)[None, :]
    ).reshape(-1)
    train_label_block = np.take(Y_target_sub, train_label_idx, axis=1)
    train_event_block = np.take(Y_event_sub, train_label_idx, axis=1)
    train_sup_block = np.take(Y_sup_sub, train_label_idx, axis=1)
    if fit_node_mask is not None:
        train_label_block = train_label_block[fit_node_mask]
        train_event_block = train_event_block[fit_node_mask]
        train_sup_block = train_sup_block[fit_node_mask]
    train_label_zero_rate = float(np.mean(train_label_block == 0))
    train_label_count = int(train_label_block.size)
    train_event_pos_count = int(np.sum(train_event_block > 0))
    train_event_neg_count = int(train_event_block.size - train_event_pos_count)
    train_sup_pos_count = int(np.sum(train_sup_block > 0))
    train_sup_neg_count = int(train_sup_block.size - train_sup_pos_count)

    # node-held-out（零历史）设定：仅保留静态/结构/外生信息；不要与表示完整历史的 L=0 混淆。
    if holdout_sub is not None and bool(holdout_sub.any()):
        ho = holdout_sub[:, None]
        Y_hist_raw_sub = np.where(ho, 0.0, Y_hist_raw_sub).astype(np.float32)
        Y_hist_target_sub = np.where(ho, 0.0, Y_hist_target_sub).astype(np.float32)
        X_hist_freq_sub = np.where(ho, 0.0, X_hist_freq_sub).astype(np.float32)
        if X_hist_freq_variants is not None:
            X_hist_freq_variants = np.where(
                holdout_sub[None, :, None], 0.0, X_hist_freq_variants
            ).astype(np.float32)
        if X_recency_sub is not None:
            X_recency_sub = np.where(ho, 0.0, X_recency_sub).astype(np.float32)

    shared_dataset_kwargs = dict(
        Y_hist_raw=Y_hist_raw_sub,
        Y_hist_target=Y_hist_target_sub,
        Y_target=Y_target_sub,
        Y_rank=Y_rank_sub,
        Y_event=Y_event_sub,
        Y_sup=Y_sup_sub,
        X_hist_freq=X_hist_freq_sub,
        X_static_public=X_static_public_sub,
        X_structural=X_structural_sub,
        X_dynamic=X_dynamic_sub,
        calendar=calendar,
        weather_by_node=weather_by_node_sub,
        hist_source_mode=graph_hist_source_mode,
        history_len=args.history_len,
        horizon=args.horizon,
    )
    ha_kwargs = dict(
        Y_hist_raw=Y_hist_raw_sub,
        Y_hist_target=Y_hist_target_sub,
        Y_target=Y_target_sub,
        Y_rank=Y_rank_sub,
        Y_event=Y_event_sub,
        hist_source_mode=ha_hist_source_mode,
        history_len=args.history_len,
        horizon=args.horizon,
    )

    graph_variant_kwargs = {}
    if X_hist_freq_variants is not None:
        graph_variant_kwargs["X_hist_freq_variants"] = X_hist_freq_variants
        graph_variant_kwargs["variant_lookbacks"] = lookback_mix
    if holdout_sub is not None:
        graph_variant_kwargs["node_holdout"] = holdout_sub
    if X_recency_sub is not None:
        graph_variant_kwargs["X_recency"] = X_recency_sub
    if budget_channel_values is not None:
        graph_variant_kwargs["budget_channel_values"] = budget_channel_values
    graph_variant_kwargs["feature_ablation"] = getattr(args, "feature_ablation", "none")
    emit_rghs = bool(getattr(args, "emit_rghs", 0)) or getattr(args, "model", "") == "st_rghs"
    if emit_rghs:
        graph_variant_kwargs["single_lookback"] = (
            hetero_lookbacks_sub if hetero_lookbacks_sub is not None else hist_prior_lookback
        )
        graph_variant_kwargs["emit_rghs"] = True
    train_graph_ds = GraphBaselineDataset(
        start_indices=train_s,
        use_priors=args.use_priors,
        mix_lookbacks=X_hist_freq_variants is not None,
        **graph_variant_kwargs,
        **shared_dataset_kwargs,
    )
    val_graph_ds = GraphBaselineDataset(
        start_indices=val_s,
        use_priors=args.use_priors,
        **graph_variant_kwargs,
        **shared_dataset_kwargs,
    )
    test_graph_ds = GraphBaselineDataset(
        start_indices=test_s,
        use_priors=args.use_priors,
        **graph_variant_kwargs,
        **shared_dataset_kwargs,
    )

    train_xgb_ds = XGBBaselineDataset(start_indices=train_s, use_priors=args.use_priors, **shared_dataset_kwargs)
    val_xgb_ds = XGBBaselineDataset(start_indices=val_s, use_priors=args.use_priors, **shared_dataset_kwargs)
    test_xgb_ds = XGBBaselineDataset(start_indices=test_s, use_priors=args.use_priors, **shared_dataset_kwargs)

    train_ha_ds = HABaselineDataset(start_indices=train_s, **ha_kwargs)
    val_ha_ds = HABaselineDataset(start_indices=val_s, **ha_kwargs)
    test_ha_ds = HABaselineDataset(start_indices=test_s, **ha_kwargs)

    A_q_sub, A_h_sub = build_diffusion_supports(edge_index_undirected_sub, idx.size, device)
    borough_id_tensor = torch.from_numpy(borough_id_sub).to(device=device)

    public_static_dim = int(X_static_public_sub.shape[1])
    structural_role_dim = int(X_structural_sub.shape[1])
    calendar_dim = int(calendar.shape[1])
    weather_dim = int(weather_by_node_sub.shape[2])
    dynamic_dim = int(X_dynamic_sub.shape[2])
    hist_channels = 2 if graph_hist_source_mode == "both" else 1
    base_graph_feature_dim = hist_channels + public_static_dim + calendar_dim + weather_dim + dynamic_dim
    graph_input_dim = base_graph_feature_dim
    if args.use_priors:
        graph_input_dim += 1
    if args.use_priors and budget_channel_values is not None:
        graph_input_dim += 1
    if args.use_priors and X_recency_sub is not None:
        graph_input_dim += 1
    if args.use_priors:
        graph_input_dim += structural_role_dim

    target_zero_rate = float(np.mean(Y_target_sub == 0))
    if graph_hist_source_mode == "target":
        hist_source_zero_rate = float(np.mean(Y_hist_target_sub == 0))
    elif graph_hist_source_mode == "both":
        hist_source_zero_rate = float(
            np.mean(
                np.concatenate(
                    [Y_hist_raw_sub[:, :, None], Y_hist_target_sub[:, :, None]],
                    axis=2,
                ) == 0
            )
        )
    else:
        hist_source_zero_rate = float(np.mean(Y_hist_raw_sub == 0))
    print(
        f"[{subgraph_label(borough_name)}] nodes={idx.size} target={target_mode} "
        f"train_Y_ZI={train_label_zero_rate:.4f} overall_Y_ZI={target_zero_rate:.4f} "
        f"hist_source={hist_source_mode} hist_channels={hist_channels} hist_Y_ZI={hist_source_zero_rate:.4f} "
        f"hist_std={hist_std:.6f} "
        f"struct={'on' if args.use_structural_prior else 'off'} "
        f"ablation={getattr(args, 'feature_ablation', 'none')} "
        f"static={'on' if getattr(args, 'use_static_features', 1) else 'off'} "
        f"win={'on' if getattr(args, 'use_event_window', 1) else 'off'} "
        f"wxcal={'on' if getattr(args, 'use_weather_calendar', 1) else 'off'} "
        f"rec={'on' if X_recency_sub is not None else 'off'} "
        f"hist={'on' if args.use_hist_freq_prior else 'off'}"
        f"{f' lookback={hist_prior_lookback}' if args.use_hist_freq_prior and hist_prior_lookback > 0 else ''}"
        f"{f' mix={lookback_mix}' if lookback_mix else ''}"
        f"{f' hetero={sorted(set(hetero_lookbacks_sub.tolist()))}' if hetero_lookbacks_sub is not None else ''} "
        f"sup={event_supervision_space} sup_ZI={1.0 - train_sup_pos_count / max(train_label_count, 1):.4f} "
        f"diffuse_target={'on' if uses_diffused_target(target_mode) else 'off'}"
        f"{f' diffuse_zero_thresh={args.risk_diffuse_zero_thresh:.4f}' if uses_diffused_target(target_mode) else ''}"
    )
    graph_loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": device.type == "cuda",
    }

    val_loaders_by_L = None
    test_loaders_by_L = None
    if X_hist_freq_variants is not None:
        val_loaders_by_L = {}
        test_loaders_by_L = {}
        for k, L in enumerate(lookback_mix):
            pinned_val = GraphBaselineDataset(
                start_indices=val_s,
                use_priors=args.use_priors,
                lookback_index=k,
                **graph_variant_kwargs,
                **shared_dataset_kwargs,
            )
            pinned_test = GraphBaselineDataset(
                start_indices=test_s,
                use_priors=args.use_priors,
                lookback_index=k,
                **graph_variant_kwargs,
                **shared_dataset_kwargs,
            )
            val_loaders_by_L[L] = DataLoader(pinned_val, shuffle=False, drop_last=False, **graph_loader_kwargs)
            test_loaders_by_L[L] = DataLoader(pinned_test, shuffle=False, drop_last=False, **graph_loader_kwargs)

    return {
        "name": borough_name,
        "num_nodes": int(idx.size),
        "target_mode": target_mode,
        "public_static_dim": public_static_dim,
        "structural_role_dim": structural_role_dim,
        "calendar_dim": calendar_dim,
        "weather_dim": weather_dim,
        "dynamic_feature_dim": dynamic_dim,
        "dynamic_feature_names": list(dynamic_feature_names),
        "dynamic_feature_stats": dynamic_feature_stats,
        "static_feature_dim": public_static_dim + structural_role_dim,
        "graph_input_dim": int(graph_input_dim),
        "hist_prior_dim": 1,
        "xgb_input_dim": int(
            hist_channels * args.history_len
            + (args.history_len if args.use_priors else 0)
            + public_static_dim
            + (structural_role_dim if args.use_priors else 0)
            + args.history_len * calendar_dim
            + args.history_len * weather_dim
            + args.history_len * dynamic_dim
        ),
        "train_graph_loader": DataLoader(
            train_graph_ds,
            shuffle=True,
            drop_last=True,
            **graph_loader_kwargs,
        ),
        "val_graph_loader": DataLoader(
            val_graph_ds,
            shuffle=False,
            drop_last=False,
            **graph_loader_kwargs,
        ),
        "test_graph_loader": DataLoader(
            test_graph_ds,
            shuffle=False,
            drop_last=False,
            **graph_loader_kwargs,
        ),
        "train_xgb_dataset": train_xgb_ds,
        "val_xgb_dataset": val_xgb_ds,
        "test_xgb_dataset": test_xgb_ds,
        "train_ha_dataset": train_ha_ds,
        "val_ha_dataset": val_ha_ds,
        "test_ha_dataset": test_ha_ds,
        "A_q": A_q_sub,
        "A_h": A_h_sub,
        "borough_id_arr": borough_id_tensor,
        "hist_mean": hist_mean,
        "hist_std": hist_std,
        "train_label_zero_rate": train_label_zero_rate,
        "train_label_count": train_label_count,
        "train_event_pos_count": train_event_pos_count,
        "train_event_neg_count": train_event_neg_count,
        "train_sup_pos_count": train_sup_pos_count,
        "train_sup_neg_count": train_sup_neg_count,
        "event_supervision_space": event_supervision_space,
        "lookback_mix": lookback_mix,
        "hetero_lookbacks": hetero_lookbacks_sub,
        "holdout_mask": holdout_sub,
        "holdout_count": int(holdout_sub.sum()) if holdout_sub is not None else 0,
        "val_loaders_by_L": val_loaders_by_L,
        "test_loaders_by_L": test_loaders_by_L,
    }
