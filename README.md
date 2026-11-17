# When Does Spatio-Temporal Learning Improve Road-Crash Screening? A History-Budget Analysis

This project studies screening of road-segment crash hotspots as a top-k
retrieval problem under a *history budget*: at each budget L, a mixed-budget
spatio-temporal model and a budget-matched, training-free crash-frequency
baseline read exactly the same left-truncated crash archive and compete for
the same screening quota. The budget-matched protocol is run on two cities
(New York and Chicago, 2021-2024). The central finding is that spatio-temporal
learning pays off only while the readable history is short: at a 14-day
budget the model reaches 0.541 Recall@20% versus 0.324 for the frequency
baseline, while at the full archive the margin shrinks to approximately zero.

This repository is the companion release for the paper of the same title
presented at ISCID 2026.

## Contents

| Path | Content |
|---|---|
| `tools/` | Evaluation, data-preparation and plotting scripts, plus chain scripts recording the exact command lines of every control run |
| `models/`, `utils/`, `main_baseline.py` | Model definitions (gated causal temporal blocks + diffusion graph convolution on a line graph), data loading, and the training entrypoint |
| `data/` | Processed New York artifacts (2021-2024): daily crash labels on 86,982 line-graph segments, line-graph geometry, link/POI/socio-demographic features, calendar and weather covariates; processing contract in `data/data.md`. Mobility expansion features are excluded (unused by the paper) |
| `data_chicago/` | Processed Chicago artifacts (43,335 harmonised segments) for the from-scratch replication |
| `analysis_results/` | Frozen result JSONs behind every table and figure number in the paper |
| `runs_results/` | Per-run evaluation and summary JSONs (no model checkpoints) |
| `supplementary.pdf` | Supplementary material of the study: extended protocol and statistical conventions (S-I), baseline/ablation tables with tie-structure and shrinkage analyses (S-II), tight-quota tables (S-III), full history-budget spectrum and endpoint details (S-IV), data-matching details (S-V) |

## Requirements

Python >= 3.10 with numpy, pandas, scipy, scikit-learn, matplotlib, pyarrow
and torch.

## Reproducing and verifying

The frozen JSONs in `analysis_results/` and `runs_results/` are the source of
truth for every numeric value reported in the paper; each can be regenerated
from the bundled processed data with the evaluation scripts, for example
`eval_cross_city_spectrum.py` (history-budget spectra per city and allocation
scope), `eval_citywide_ranker_variants.py` (citywide ranking-convention
sensitivity), `eval_simple_supervised_baselines.py` (static and fusion
baselines), `bootstrap_margin_ci.py` (paired block-bootstrap confidence
intervals) and `eval_generalist_at_L.py` (per-budget checkpoint evaluation).
The exact command lines of every control run (strong simple baselines,
single-budget endpoint models, no-graph ablation, input-ablation retrains,
Chicago and New York multi-seed runs) are recorded in the `tools/*.sh` chain
scripts (`run_review_revision_chain.sh`, `run_review3_chain.sh`,
`run_feature_source_chain.sh`). On shared machines, GPU jobs should be
serialised (for example through a flock-based wrapper) so that only one
training or evaluation job uses the GPU at a time.

Regenerating the figures:

```bash
python3 tools/plot_iscid_spectrum.py              # 2x2 history-budget spectrum
python3 tools/plot_iscid_spectrum.py margins1col  # single-column margin panels
python3 tools/plot_budget_protocol.py             # budget-matched protocol figure
```

The training entrypoint is `main_baseline.py` (mixed-budget sampling: the
budget is drawn uniformly per training window from {14, 30, 90, 180, full}).
The exact command lines for the control runs (no-graph ablation,
single-budget endpoint models, Chicago and New York multi-seed runs) are
recorded in the `tools/*.sh` chain scripts. On shared machines, GPU jobs
should be serialised (for example through a flock-based wrapper) so that only
one training or evaluation job uses the GPU at a time.

## Data

All raw inputs are public and cited in the paper: NYC OpenData Motor Vehicle
Collisions, City of Chicago Traffic Crashes, OpenStreetMap, and the U.S.
Census 2020 American Community Survey. This repository distributes processed
derivatives only; `data/data.md` documents the processing contract (key
fields, event-score reconstruction, static-feature scaling).

## Notes

- No model checkpoints are distributed. The mixed-budget model has 155,331
  parameters and retrains from scratch (seed 42; seeds 43/44 for the
  multi-seed cells).
- Reported shares are ranking fractions of screened segments, not inspection
  counts or measured safety benefits; the paper claims no measured safety
  impact.
- Code, data and results are released under the MIT License (see `LICENSE`).

## Citation

"When Does Spatio-Temporal Learning Improve Road-Crash Screening? A
History-Budget Analysis." ISCID 2026.
