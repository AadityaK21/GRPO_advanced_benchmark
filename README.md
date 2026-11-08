# GRPO advantage benchmark

Published GRPO advantage estimators, compared under matched conditions: one training loop, one task
(GSM8K), the same rollout budget and tuning budget for every method, ten seeds, bootstrap intervals.

## The falsification rule

Written before any benchmark run. Do not change it after seeing results.

> A method is counted as beating vanilla GRPO only if its paired bootstrap test against vanilla at group
> size 8, on the primary metric, rejects at 5% after Holm's correction across the nine Tier 1 methods
> (2-GRPO runs at its own group size of two). Results at other group sizes, and for Tier 2, are reported
> with intervals and labelled exploratory. Methods that beat vanilla at one group size and not others are
> reported as such, not as winners.

With nine tests at 5% each, at least one false winner turns up more than a third of the time even if
nothing works. That is why the correction is there.

**Test details.** Fix these before Gate 1, then leave them alone:

- Primary metric: exact-match accuracy on the full GSM8K test split (1319 problems), greedy decoding,
  **final checkpoint**. The checkpoint chosen on validation is reported too, but it is not the test.
- Statistic: interquartile mean (IQM) over 10 seeds.
- Test: paired by seed (seed k of the method against seed k of vanilla). 50,000 bootstrap resamples of
  the seeds, the same resample for both. Two-sided p-value. Holm at 5% across the nine Tier 1 methods.
- Code: `src/eval/stats.py`, run by `scripts/aggregate.py`. Checked by `tests/test_null.py`.

## Running

```
pip install torch==2.14.1 transformers==5.19.0 trl==1.15.0 datasets peft pyyaml pytest scikit-learn matplotlib scipy
pytest                                                           # all tests, CPU, under a minute

python scripts/run.py --method vanilla --seed 0 --smoke          # whole pipeline in a few minutes
python scripts/smoke_all.py                                      # every method, smoke size, one table
                                                                 # (on Kaggle: notebooks/kaggle_smoke_all.ipynb)
python scripts/run.py --method vanilla --seed 0                  # one benchmark run
python scripts/run.py --method dr_grpo --seed 0 --set budget.group_size=4
python scripts/run.py --method two_grpo --seed 0 --set budget.group_size=2   # 2-GRPO runs only at G = 2
python scripts/run.py --eval-only --seed 0                       # the untrained base model
python scripts/aggregate.py                                      # IQM table and the falsification test

python scripts/tune.py plan                                      # the tuning trials, nothing runs
python scripts/tune.py run                                       # run unfinished trials (validation only)
python scripts/tune.py collect                                   # best trial per method -> configs/tuned.yaml
python scripts/grid.py status                                    # the benchmark grid (configs/grid.yaml)
python scripts/grid.py worker --block tier1                      # run pending jobs (HPC: hpc/grid_worker.pbs)
python scripts/figures.py                                        # every figure and table from results/
```

Each test-split run also reports pass@1/4/16 of the final weights (16 sampled completions per problem):
a secondary metric, outside the falsification rule.

Tier 1 methods (`--method`): `vanilla`, `dr_grpo`, `dapo`, `rloo`, `sign`, `ngrpo`, `softmax_grpo`, `corpo`,
`mc_grpo`, `two_grpo`. Tier 2: `ebpo`, `krpo`, `bv_blend`, `ha_dw`, `signbalance`, `lens`. What each paper
specifies: `docs/method-specs.md` and `docs/method-specs-tier2.md`. `bv_blend` needs prompt clusters, made
once with `python scripts/make_clusters.py` (sentence-transformers, scikit-learn). Their free parameters (`softmax_grpo` tau, `corpo` c) are tuned with
`--set advantage.params.tau=0.3` and the like, inside the equal tuning budget.

Each run writes `results/<method>/g<group size>/seed<seed>-<time>/`: the full config, the seed, the git
commit, a line per training step, every validation check, and `summary.json` with the test accuracy.
Commit before a benchmark run: a run from a tree with uncommitted changes is flagged by `aggregate.py`.

## Layout

- `configs/base.yaml`: everything every condition shares. `configs/methods/<name>.yaml`: one per method,
  only its `advantage`, `loss` and `filter` sections (enforced). `filter` is DAPO's dynamic sampling.
- `src/advantages/`: one file per method, all with the same signature (see `base.py`).
- `src/train/`: the GRPO loop. Nothing method-specific. `logprobs.py` scores the output layer in chunks.
- `src/reward/`: the GSM8K answer parser and exact-match reward.
- `src/data.py`: the fixed validation split and few-shot examples. `src/eval/`: accuracy, pass@k and statistics.
- `scripts/run.py`: one run. `scripts/aggregate.py`: all runs. `scripts/tune.py`: the tuning sweep
  (`configs/tuning.yaml`). `scripts/grid.py`: every benchmark run (`configs/grid.yaml`, `hpc/`).
  `scripts/figures.py`: the paper's figures and tables. `scripts/train_grpo.py` and
  `scripts/train_trl.py`: the week-7 check that this loop matches TRL.
- `docs/implementation-notes.md`: every choice a paper leaves open, and the verification log.
  `docs/compute.md`: measured time per run and the grid that fits.
- `warmup/`: REINFORCE and PPO from scratch (weeks 1–4). Each has its own tests.
