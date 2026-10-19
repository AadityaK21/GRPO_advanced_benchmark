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
pip install torch==2.14.1 transformers==5.19.0 trl==1.15.0 datasets peft pyyaml pytest
pytest                                                           # all tests, CPU, under a minute

python scripts/run.py --method vanilla --seed 0 --smoke          # whole pipeline in a few minutes
python scripts/run.py --method vanilla --seed 0                  # one benchmark run
python scripts/run.py --method dr_grpo --seed 0 --set budget.group_size=4
python scripts/run.py --eval-only --seed 0                       # the untrained base model
python scripts/aggregate.py                                      # IQM table and the falsification test
```

Each run writes `results/<method>/g<group size>/seed<seed>-<time>/`: the full config, the seed, the git
commit, a line per training step, every validation check, and `summary.json` with the test accuracy.
Commit before a benchmark run: a run from a tree with uncommitted changes is flagged by `aggregate.py`.

## Layout

- `configs/base.yaml`: everything every condition shares. `configs/methods/<name>.yaml`: one per method,
  only its `advantage` and `loss` sections (enforced).
- `src/advantages/`: one file per method, all with the same signature (see `base.py`).
- `src/train/`: the GRPO loop. Nothing method-specific. `logprobs.py` scores the output layer in chunks.
- `src/reward/`: the GSM8K answer parser and exact-match reward.
- `src/data.py`: the fixed validation split and few-shot examples. `src/eval/`: accuracy and statistics.
- `scripts/run.py`: one run. `scripts/aggregate.py`: all runs. `scripts/train_grpo.py` and
  `scripts/train_trl.py`: the week-7 check that this loop matches TRL.
- `docs/implementation-notes.md`: every choice a paper leaves open, and the verification log.
  `docs/compute.md`: measured time per run and the grid that fits.
- `warmup/`: REINFORCE and PPO from scratch (weeks 1–4). Each has its own tests.
