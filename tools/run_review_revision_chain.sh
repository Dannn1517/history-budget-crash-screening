#!/usr/bin/env bash
# Experiment chain for the strong simple baselines and single-budget
# endpoint models: XGBoost full-feature baselines, single-budget generalist
# training at L=14 and full, 9-budget evaluation grid, and a sanity
# re-evaluation of the mixed-budget checkpoint against the frozen table.
#
# Serial GPU jobs via a flock-based serialisation wrapper; aborts on first
# failure.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
GPU_RUN="${GPU_RUN:?set GPU_RUN to your flock-based GPU serialisation wrapper}"
LOG="${ROOT}/runs_results/review_controls/review_revision_chain.log"

cd "${ROOT}" || exit 1
mkdir -p runs_results/review_controls

run_job() {
    local name="$1"; shift
    echo "=== [${name}] start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"${LOG}"
    env PYTHONPATH="${ROOT}" "${GPU_RUN}" "$@" >>"${LOG}" 2>&1
    local rc=$?
    echo "=== [${name}] exit=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"${LOG}"
    return ${rc}
}

XGB_ARGS=(--model xgboost --selection_metric es_ndcg --hr_ratio 0.05,0.1,0.2 --target_mode risk --event-cumulative-horizons "")
TRAIN_ARGS=(--model st_event_occurrence --seed 42 --selection_metric es_ndcg --hr_ratio 0.05,0.1,0.2 --target_mode risk --event-cumulative-horizons "")

run_job "xgb_L14" "${PY}" main_baseline.py --out_dir runs_results/review_controls/xgboost_L14_rs --hist_prior_lookback 14 "${XGB_ARGS[@]}" || exit 1
run_job "xgb_L90" "${PY}" main_baseline.py --out_dir runs_results/review_controls/xgboost_L90_rs --hist_prior_lookback 90 "${XGB_ARGS[@]}" || exit 1

run_job "single_L14" "${PY}" main_baseline.py --out_dir runs_results/review_controls/single_budget_L14 --hist_prior_lookback 14 "${TRAIN_ARGS[@]}" || exit 1
run_job "single_full" "${PY}" main_baseline.py --out_dir runs_results/review_controls/single_budget_full --hist_prior_lookback 0 "${TRAIN_ARGS[@]}" || exit 1

BUDGETS="14 21 30 60 90 120 180 365 0"
for L in ${BUDGETS}; do
    run_job "eval_single_L14_L${L}" "${PY}" tools/eval_generalist_at_L.py \
        --checkpoint runs_results/review_controls/single_budget_L14/st_event_occurrence_weighted_bce_best.pt \
        --lookback "${L}" --split test --out "runs_results/review_controls/single_budget_L14/eval_L${L}.json" || exit 1
done
for L in ${BUDGETS}; do
    run_job "eval_single_full_L${L}" "${PY}" tools/eval_generalist_at_L.py \
        --checkpoint runs_results/review_controls/single_budget_full/st_event_occurrence_weighted_bce_best.pt \
        --lookback "${L}" --split test --out "runs_results/review_controls/single_budget_full/eval_L${L}.json" || exit 1
done

MIXED_CKPT="runs_results/history_adaptive_formal/generalist_6ep_shared_stats_rs/st_event_occurrence_weighted_bce_best.pt"
for L in 14 90 0; do
    run_job "eval_mixed_L${L}" "${PY}" tools/eval_generalist_at_L.py \
        --checkpoint "${MIXED_CKPT}" --lookback "${L}" --split test \
        --out "runs_results/review_controls/mixed_sanity_eval_L${L}.json" || exit 1
done

echo "=== [CHAIN] all jobs done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"${LOG}"
