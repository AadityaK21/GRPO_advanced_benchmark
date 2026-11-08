# Compute

Measured, not estimated. Fill in from `summary.json` of real runs (`train_seconds`, `eval_seconds`,
`gpu_peak_gb`) and `meta.json` (`gpu`, `dtype`): `python scripts/compute_report.py` prints both tables
below, filled in, from every finished run.

## Time per run

| Date | GPU | dtype | Model | G | Rollouts | Train min | Eval min | Total min | Peak GB | Run folder |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

## The grid

Runs needed (from the plan): Tier 1 at G = 4, 8, 16 × 10 seeds, 2-GRPO at G = 2, null control, base
model, Tier 2 at G = 8, then the 1.5B check. About 470 runs plus tuning trials and re-runs.

GPU-hours = minutes per run × runs / 60, then double it for tuning and re-runs.

| Block | Runs | Minutes / run | GPU-hours | ×2 |
|---|---|---|---|---|
| Tier 1, 0.5B | | | | |
| Null control and base | | | | |
| Tier 2, 0.5B | | | | |
| 1.5B check | | | | |
| **Total** | | | | |

If the total does not fit the HPC share for weeks 15–24, cut in the plan's order: the optional build
phase, then Tier 2, then the 1.5B check, then group sizes 4 and 16. Never seeds.

## What else costs GPU time

- **pass@k** (`eval.passk_samples` 16 on all 1319 test problems): 21,104 sampled completions per run,
  about 1.6 times the 12,800 training rollouts, but sampling only (no backward pass). Measure it in the
  timing run (`eval_seconds` in `summary.json` includes it). If it does not fit, set
  `eval.passk_problems` to a fixed prefix for every run, and record it in the decision log.
- **Tuning** (`configs/tuning.yaml`): 16 methods x 8 trials x 3200 rollouts = 128 runs at a quarter
  length, validation only (no test, no pass@k): about 32 full runs, plus their validation passes.
- **Grid sizes** (`python scripts/grid.py status`): gate1 11, tier1 281, null_control 10, scale_1p5b 101,
  tier2 60.
