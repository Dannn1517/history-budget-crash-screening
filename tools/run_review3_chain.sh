#!/usr/bin/env bash
# Control-experiment chain: attribution ablation (no-graph retrain) and
# multi-seed citywide spectra.
#
# Serial GPU jobs via a flock-based serialisation wrapper. Exit 75 (GPU occupied)
# is retried with backoff; any other non-zero aborts the chain.
# Every job is idempotent: it is skipped when its output artefact exists, so
# the chain can be relaunched safely after a failure.
#
# Jobs: s42 citywide rebuild consistency check; no-graph training and evals;
# Chicago training (seeds 43/44) and citywide spectra; NYC s43/44 spectra.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
GPU_RUN="${GPU_RUN:?set GPU_RUN to your flock-based GPU serialisation wrapper}"
LOG="${ROOT}/runs_results/review_controls/review3_chain.log"

cd "${ROOT}" || exit 1
mkdir -p runs_results/review_controls

run_job() {
    local name="$1"; shift
    local rc=75 attempt
    for attempt in 1 2 3 4 5 6 7 8; do
        echo "=== [${name}] start $(date -u +%Y-%m-%dT%H:%M:%SZ) attempt=${attempt} ===" >>"${LOG}"
        env PYTHONPATH="${ROOT}" "${GPU_RUN}" "$@" >>"${LOG}" 2>&1
        rc=$?
        echo "=== [${name}] exit=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"${LOG}"
        [ "${rc}" -ne 75 ] && break
        echo "=== [${name}] GPU occupied elsewhere; sleeping 600s before retry ===" >>"${LOG}"
        sleep 600
    done
    return ${rc}
}

run_unless_exists() {
    local name="$1"; local artefact="$2"; shift 2
    if [ -f "${artefact}" ]; then
        echo "=== [${name}] skipped ($(basename "${artefact}") exists) ===" >>"${LOG}"
        return 0
    fi
    run_job "${name}" "$@"
}

NY_HARSF=runs_results/history_adaptive_formal/safe_fallback_fixed_switch_seed42_clean_rs/st_rghs_best.pt
CHI_HARSF=runs_results/chicago_formal/safe_fallback_fixed_switch/st_rghs_best.pt
NG_DIR=runs_results/review_controls/no_graph_mixed_s42
NG_CKPT="${NG_DIR}/st_event_occurrence_weighted_bce_best.pt"

MIX_ARGS=(--model st_event_occurrence --event_loss weighted_bce --mask_borough all
          --target_mode risk --hist_prior_lookback_mix 14,30,90,180,0
          --hist_prior_shared_stats 1 --epochs 6 --repeat_runs 1
          --selection_metric es_ndcg --hr_ratio 0.05,0.1,0.2
          --event-cumulative-horizons "" --progress_interval 500)
CHI_ARGS=(--model st_event_occurrence --event_loss weighted_bce --mask_borough all
          --target_mode risk --hist_prior_lookback_mix 14,30,90,180,0
          --hist_prior_shared_stats 1 --epochs 6 --repeat_runs 1
          --selection_metric es_ndcg --event-cumulative-horizons ""
          --progress_interval 500)

# ---- s42 citywide recompute (rebuild-path consistency check) ----
run_unless_exists "nyc_s42_citywide_recheck" \
    "runs_results/review_controls/review3_controls/nyc_s42_citywide_recheck.json" \
    "${PY}" tools/eval_cross_city_spectrum.py \
    --harsf-checkpoint "${NY_HARSF}" \
    --generalist-checkpoint runs_results/history_adaptive_formal/generalist_6ep_shared_stats_rs/st_event_occurrence_weighted_bce_best.pt \
    --scope citywide --budgets 14,90,0 \
    --out runs_results/review_controls/review3_controls/nyc_s42_citywide_recheck.json || exit 1

# ---- no-graph training (NYC, seed 42) ----
run_unless_exists "no_graph_train" "${NG_CKPT}" \
    "${PY}" main_baseline.py --seed 42 --diffusion_steps 0 \
    --out_dir "${NG_DIR}" "${MIX_ARGS[@]}" || exit 1

# ---- no-graph stratified evals ----
for L in 14 90 0; do
    run_unless_exists "no_graph_eval_L${L}" "${NG_DIR}/eval_L${L}.json" \
        "${PY}" tools/eval_generalist_at_L.py \
        --checkpoint "${NG_CKPT}" --lookback "${L}" --split test \
        --out "${NG_DIR}/eval_L${L}.json" || exit 1
done

# ---- no-graph citywide spectrum (raw primary convention) ----
run_unless_exists "no_graph_citywide" "${NG_DIR}/citywide_spectrum.json" \
    "${PY}" tools/eval_cross_city_spectrum.py \
    --harsf-checkpoint "${NY_HARSF}" \
    --generalist-checkpoint "${NG_CKPT}" --scope citywide --budgets 14,90,0 \
    --out "${NG_DIR}/citywide_spectrum.json" || exit 1

# ---- Chicago seeds 43/44 ----
for S in 43 44; do
    CHI_DIR="runs_results/chicago_formal/generalist_s${S}"
    run_unless_exists "chicago_train_s${S}" "${CHI_DIR}/st_event_occurrence_weighted_bce_best.pt" \
        "${PY}" main_baseline.py --data_dir data_chicago \
        --seed "${S}" --out_dir "${CHI_DIR}" "${CHI_ARGS[@]}" || exit 1
done

# ---- Chicago seeds 43/44 citywide spectra ----
for S in 43 44; do
    CHI_DIR="runs_results/chicago_formal/generalist_s${S}"
    run_unless_exists "chicago_s${S}_citywide" "${CHI_DIR}/citywide_spectrum.json" \
        "${PY}" tools/eval_cross_city_spectrum.py \
        --harsf-checkpoint "${CHI_HARSF}" \
        --generalist-checkpoint "${CHI_DIR}/st_event_occurrence_weighted_bce_best.pt" \
        --scope citywide --budgets 14,90,0 \
        --out "${CHI_DIR}/citywide_spectrum.json" || exit 1
done

# ---- NYC seeds 43/44 citywide spectra (existing checkpoints) ----
for S in 43 44; do
    NY_DIR="runs_results/feature_source/generalist_s${S}"
    run_unless_exists "nyc_s${S}_citywide" "${NY_DIR}/citywide_spectrum.json" \
        "${PY}" tools/eval_cross_city_spectrum.py \
        --harsf-checkpoint "${NY_HARSF}" \
        --generalist-checkpoint "${NY_DIR}/st_event_occurrence_weighted_bce_best.pt" \
        --scope citywide --budgets 14,90,0 \
        --out "${NY_DIR}/citywide_spectrum.json" || exit 1
done

echo "=== chain complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"${LOG}"
