from __future__ import annotations


COUNT_LIKE_MODELS = {
    "stnb_gnn",
    "stzinb",
}


def normalize_target_mode(target_mode: str) -> str:
    mode = str(target_mode).strip().lower()
    if mode not in {"risk", "risk_diffuse"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return mode


def normalize_hist_source_mode(hist_source_mode: str) -> str:
    mode = str(hist_source_mode).strip().lower()
    if mode not in {"raw_risk", "target", "both"}:
        raise ValueError(f"Unsupported hist_source_mode: {hist_source_mode}")
    return mode


def uses_diffused_target(target_mode: str) -> bool:
    return normalize_target_mode(target_mode) == "risk_diffuse"


def should_warn_distribution_mismatch(model_name: str, target_mode: str) -> bool:
    return uses_diffused_target(target_mode) and model_name in COUNT_LIKE_MODELS
