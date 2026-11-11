# Project 1 handoff: GRPO advantage benchmark

Written 28 Oct 2026 for anyone, human or LLM, who picks up this repo. Read this file first, then
`README.md`, then the files it points to. Updated 8 Nov 2026: the tuning sweep, pass@k, the HPC grid
and the figure script now exist (code and tests only; none has run on a GPU yet). Everything below is
checked against commit 242cdec plus the files listed under "Uncommitted at the time of writing".

---

## 1. The project in one paragraph

GRPO (Group Relative Policy Optimization) is the RL algorithm used to post-train LLMs on tasks with a
checkable answer. For each prompt it samples a group of G answers, scores them, and turns the scores
into "advantages": how much each answer is pushed up or down. Since 2024 more than a dozen papers have
proposed a better advantage estimator, each tested on its own model, data, budget and seed count, so
their claims cannot be compared. This project implements 15 of them plus vanilla GRPO in ONE training
loop and compares them under matched conditions: same model, same task, same rollout budget, same
tuning budget, ten seeds, bootstrap intervals, and a pass/fail rule written before any run. The
expected outcome is honest either way: a practical guide if some methods really win, or a clear null
("indistinguishable from seed noise at this scale") if none do.

- **Owner:** Aaditya Kumawat, 3rd-year B.Tech, IIT Delhi. GitHub `AadityaK21`, repo
  `AadityaK21/GRPO_advanced_benchmark` (private). New to RL and mainstream ML; strong in C++ and
  systems. About 20 hours a week.
- **Target:** a workshop paper at ICML 2027 (deadline around May 2027), then TMLR. Acceptance is not
  assumed anywhere.
- **The full plan** (roadmap, protocol, paper skeleton, failure modes) is the doc "Project 1 — GRPO
  Advantage Benchmark" (claude.ai artifact 1DoF6ogcFjsnxX5WeUuV9U). Section 9 below summarises it.

## 2. How to work with the owner

- Explain in simple English, step by step. Short answers. He is learning, so say WHY, not only what.
- He commits and pushes himself, from PowerShell on Windows (`C:\GRPO`). Give him the exact commands.
  Never push for him.
- The GitHub token lives only in Kaggle Secrets as `GITHUB_TOKEN`. Never print it, log it or write it
  into a file or notebook output.
- Every decision that could change a number goes into `docs/implementation-notes.md` (decision log)
  BEFORE the run that uses it.

## 3. Where things stand (8 Nov 2026)

**Done and verified**

| What | Evidence |
|---|---|
| Own GRPO loop (sampling, reward, advantages, clipped loss, KL, micro-batching) | `src/train/` |
| Loop matches TRL 1.15.0 | Same reward and completion length on 100 of 100 steps, Qwen2.5-0.5B-Instruct, GSM8K, LoRA r16, lr 1e-5, G 8, seed 0, Kaggle T4. Control with lr 0 differs from step 1, so the match depends on the updates. `tests/test_against_trl.py` 53/53 on Kaggle |
| Memory fix: output layer scored 256 tokens at a time | Peak GPU 11.7 GB -> 3.19 GB (TRL: 3.2 GB), log-probs identical (20/20 steps) |
| Harness: configs, run logging, evaluation, aggregation, statistics | `scripts/run.py --smoke` ran end to end on a Kaggle T4 in 1.6 min, peak 3.1 GB |
| Statistics checked under the null (simulated) | Unpaired IQM intervals exclude 0 in 4.6% of null splits; the paired test rejects 6.0-6.5% (slightly above 5%: the real null control matters) |
| Falsification rule written in `README.md` | Written before any benchmark run. Do not edit it |
| All 16 methods implemented, each from its paper's equations | Unit tests reproduce every hand-worked example in `docs/method-specs.md` and `docs/method-specs-tier2.md` to 4 decimals |
| Tuning sweep, pass@k, HPC grid, figures, compute report (8 Nov) | `scripts/tune.py`, `src/eval/passk.py`, `scripts/grid.py` + `hpc/`, `scripts/figures.py`, `scripts/compute_report.py`. Tested on the tiny model and on fake results, then checked by an independent review agent; its 9 findings are fixed and each has a test. Never run on a GPU |
| Test suite | 273 passed, 1 skipped on CPU (the skip is the Triton-kernel TRL check, which runs on a GPU) |

**Not done yet (in the order it should happen: see section 8)**

1. ~~Smoke-test all 16 methods on a real GPU~~ DONE 11 Nov: all 17 configs pass on a Kaggle T4
   (verification log).
2. ~~Make the BV-Blend cluster file~~ DONE 11 Nov, committed (`data/gsm8k_train_clusters_k16.json`).
3. ~~The prompt check~~ DONE 11 Nov: 3-shot is final (0.596 vs 0.550 zero-shot on validation; decision log).
4. The timing run (one full vanilla run, evaluation included) and the compute budget in `docs/compute.md`.
5. IIT Delhi HPC access (A100-class GPUs, bf16). Not requested/confirmed yet.
6. Gate 1: vanilla (10 seeds) vs the base model.
7. Missing code: only the optional mid-run checkpoint/resume (section 10). Everything else exists.
8. Author emails for ambiguous papers (MC-GRPO first).
9. A human re-read of the six Tier 2 papers (their specs were extracted through a web-fetch tool).

**Commits so far (newest first)**
`242cdec` eval progress lines, device check, `--val-only` · `d93356c` null control, handoff ·
`83cd360` Tier 2 methods, smoke_all, clusters script · `0d21b70` Tier 1 methods, DAPO dynamic sampling ·
`a027dd0` method specs, truncation rule, diagnostics · `1a092be` verification log · `c012536` harness,
advantage interface, aggregation, Dr. GRPO · `027152c` chunked log-probs.

**Uncommitted at the time of writing (8 Nov):** new `src/eval/passk.py`, `scripts/tune.py`,
`scripts/grid.py`, `scripts/figures.py`, `scripts/compute_report.py`, `configs/tuning.yaml`, `configs/grid.yaml`, `hpc/` (3 files),
`tests/test_passk.py`, `tests/test_tune.py`, `tests/test_grid.py`, `tests/test_figures.py`,
`tests/test_compute_report.py`; changed
`configs/base.yaml` (eval.at_start, eval.passk_*), `scripts/run.py`, `scripts/aggregate.py`,
`tests/test_run_script.py`, `tests/test_aggregate.py`, `.gitignore`, `README.md`, this file,
`docs/implementation-notes.md`, `docs/compute.md`.

## 4. Repo map

```
README.md                 What the project is, the falsification rule, how to run
pyproject.toml, pytest.ini
configs/
  base.yaml               EVERYTHING every condition shares (model, LoRA, optimiser, data, prompt,
                          sampling, reward, budget, filter, advantage, loss, eval). Values marked
                          PROVISIONAL are frozen after the timing run and Gate 1
  methods/<name>.yaml     One per method. May set ONLY the `advantage`, `loss` and `filter` sections
                          (src/config.py enforces it). 17 files incl. vanilla_null
  tuning.yaml             The tuning budget: trials per method, seeds, trial length, lr range, params
  grid.yaml               Every benchmark run, block by block (gate1, tier1, null_control, scale_1p5b, tier2)
  tuned.yaml              Written by `tune.py collect` (not yet: no tuning has run). Commit it
hpc/                      grid_worker.pbs (IIT Delhi, PBS), grid_array.slurm, setup_offline.sh
src/
  config.py               load_config(method, overrides): base.yaml + method file + --set overrides.
                          Unknown keys are errors (typos never pass silently)
  data.py                 GSM8K loading, the fixed val/shots/train split (split_seed 1234), few-shot prompt
  reward/gsm8k.py         Answer parser and exact-match reward
  advantages/
    base.py               The method interface: AdvantageBatch, Advantage, build_advantage
    grpo.py               group_advantages(): the shared z-score helper (sample std, +1e-4)
    <method>.py           One file per method (16), each with a docstring: paper, equation, choices
  train/
    sample.py             sample_group(): G completions per prompt, plain temperature sampling
    logprobs.py           Per-token log-probs, output layer in 256-token chunks (the memory fix)
    loss.py               grpo_loss(): clipped surrogate, k3 KL, 4 aggregation types
    loop.py               grpo_step(): one full step. Dynamic sampling refill (DAPO) lives here
    model.py              load_policy(): model, tokenizer, LoRA, dtype
  eval/accuracy.py        Greedy evaluation: accuracy, format rate, per-problem correctness
  eval/passk.py           pass@k: n sampled completions per test problem, unbiased estimator (Chen et al.)
  eval/stats.py           IQM, percentile bootstrap, paired bootstrap, Holm, null control
  runlog.py               Run folders, config/seed/commit/meta writing
scripts/
  run.py                  ONE run: train under a rollout budget, validate, test. The main entry point
  smoke_all.py            Every method at smoke size, one table (GPU check, not a result)
  make_clusters.py        Prompt clusters for BV-Blend (sentence-transformers + k-means)
  aggregate.py            All runs -> IQM table with intervals + the falsification test + null control,
                          plus pass@k (secondary, exploratory: outside the Holm family)
  tune.py                 The equal-budget tuning sweep: plan / run / collect -> configs/tuned.yaml
  grid.py                 The benchmark grid from configs/grid.yaml: status / list / worker / run-one.
                          Resumable, claims jobs so workers never collide, refuses uncommitted code
  figures.py              Every paper figure (PDF, PNG, CSV) and table (md, tex) from results/
  compute_report.py       After the timing run: measured minutes per run -> GPU-hours of the whole plan
  train_grpo.py, train_trl.py, compare_runs.py   The week-7 "our loop vs TRL" check
notebooks/kaggle_smoke_all.ipynb   Clone (token from Kaggle Secrets), clusters, smoke_all
docs/
  HANDOFF.md              This file
  implementation-notes.md Every open choice and what we chose; status table; verification log;
                          decision log; open questions; how to ask authors
  method-specs.md         Tier 1: what each paper actually specifies, with worked examples
  method-specs-tier2.md   Tier 2: same, from six research agents (see the caveat at its top)
  compute.md              Time per run and the grid (EMPTY: filled by the timing run)
  grpo-loop.md            How the loop and its tests are organised; laptop setup
tests/                    273 tests, CPU only, offline. tiny.py builds a 2-layer Qwen2 for them
warmup/                   REINFORCE (CartPole) and PPO from scratch, weeks 1-4 (on the laptop copy)
results/, runs/           Run outputs (git-ignored)
```

## 5. The shared setup (configs/base.yaml) and why

| Setting | Value | Note |
|---|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct; 1.5B-Instruct later as the scale check | `--set model.name=...` |
| Fine-tuning | LoRA r 16, alpha 32, dropout 0, all linear layers; gradient checkpointing | dropout 0 keeps log-probs deterministic |
| dtype | `auto`: bf16 where native, else fp32 | Kaggle T4 = fp32 = development only. All benchmark runs must share one GPU type and dtype |
| Optimiser | AdamW, lr 1e-5 PROVISIONAL, linear decay to 0, no warmup, grad clip 1.0 | Each method's lr comes from the tuning sweep |
| Data | GSM8K train shuffled with split_seed 1234: first 500 = validation, next 3 = few-shot examples, rest (6,970) = training pool. Test = all 1,319 | The run seed never touches the split |
| Prompt | 3-shot chat turns + instruction `Let's think step by step and output the final answer after "####".` | Final: the prompt check (11 Nov) measured 3-shot 0.596 vs zero-shot 0.550 on validation, paired CI [+0.002, +0.090] |
| Sampling | temperature 1.0, max 512 new tokens, no top-k/top-p | log-probs assume exactly this distribution |
| Reward | 1 if the parsed number equals the reference, else 0. Truncated (no EOS by 512 tokens) = 0 | Dr. GRPO's code does the same |
| Budget | 64 rollouts per step (prompts per step = 64 // G), 12,800 per run = 200 steps at G 8. PROVISIONAL | Fixed ROLLOUTS, not steps. Extra rollouts (MC-GRPO) and refills (DAPO) count |
| Loss | `grpo` aggregation (per-completion token mean, then mean), clip 0.2/0.2, beta 0 (no KL), 1 update per batch | With 1 update the ratio is 1 and the clip never acts |
| Advantage std | sample std (n - 1) + 1e-4 | As TRL and every reference implementation |
| Evaluation | greedy, 512 tokens, validation every 50 steps and at step 0; the best-validation weights are also tested | Primary metric = FINAL checkpoint (README) |

The rule for method configs: a method gets a loss or sampling change only when that change is part of
what its paper proposes (DAPO's clip, loss and dynamic sampling; NGRPO's asymmetric clip; Dr. GRPO's and
RLOO's sequence-sum loss). Framework defaults the authors happened to run are recorded as deviations,
not copied.

## 6. The 16 methods

`G` = group size. "PROV" = a value the paper does not give, chosen by us, to be tuned or checked.

| # | `--method` | Paper (arXiv) | What it does (0/1 rewards) | Own settings | Special |
|---|---|---|---|---|---|
| 0 | `vanilla` | 2402.03300 | (r - mean)/(std + 1e-4) | - | the baseline |
| 1 | `dr_grpo` | 2503.20783 | r - mean, no std; loss sum / (n x 512) | loss `dr_grpo` | loss ~100x smaller: lr sweep must cover it |
| 2 | `dapo` | 2503.14476 | vanilla + overlong penalty (L_cache 102, x0.5); drops all-equal groups and refills | clip 0.2/0.28, loss `dapo`, `filter.drop_uniform_groups` | every refill counts; lr decays by rollouts spent |
| 3 | `rloo` | 2402.14740 | leave-one-out baseline = G/(G-1) x Dr. GRPO | loss `dr_grpo` | expect = Dr. GRPO within noise |
| 4 | `sign` | 2605.07689 | 2r - 1: +1 right, -1 wrong, whatever the group | - | all-wrong groups get -1 |
| 5 | `ngrpo` | 2509.18851 | a virtual reward-1 sample in every group's mean and std | clip 0.16/0.24 | all-wrong groups pushed down |
| 6 | `softmax_grpo` | 2608.09271 | G softmax(r/tau) - 1 | tau 0.1 (tune {0.1, 0.3}) | paper picked tau on test |
| 7 | `corpo` | 2511.04439 | r - max(c, mean) | c 0.5 PROV (tune {0.25, 0.5, 0.75}) | c = 0 is exactly Dr. GRPO |
| 8 | `mc_grpo` | 2601.22582 | G+1 samples, median/MAD; drop the median one | eps 1e-3 | MAD = 0 on 0/1 rewards: minority answers get +-1000, majority 0 |
| 9 | `two_grpo` | 2510.00977 | vanilla at G = 2 only | - | run with `--set budget.group_size=2`; refuses other G |
| 10 | `ebpo` | 2602.05165 | baseline shrunk to the global mean by a running empirical-Bayes S; batch-normalised | - | state: 6 running scalars |
| 11 | `krpo` | 2505.07527 | Kalman filter over the group in row order, reset per group | Q 1e-5, R 1e-2 | advantages ~10x vanilla's; order-dependent |
| 12 | `bv_blend` | 2606.28707 | baseline/scale blended with per-cluster history by certainty | T 0.1, N0 8, V_prior 0.25, delta_N 1 PROV | needs `data/gsm8k_train_clusters_k16.json` |
| 13 | `ha_dw` | 2601.08521 | vanilla x lambda exp(-sgn(A)(p_hat - C)), C a running accuracy anchor | lambda 1.3, eta 10 PROV, m 10 | eta 0.1 freezes the anchor |
| 14 | `signbalance` | 2609.04063 | +1 right, -n+/n- wrong, per group | all-right groups +1 (Algorithm 2) | `all_correct=zero` = sensitivity check |
| 15 | `lens` | 2510.08696 | vanilla on a reward that penalises confident wrong answers more; all-wrong groups x alpha | alpha 0.25 | needs seq log-probs (loop provides them) |
| - | `vanilla_null` | - | exactly vanilla, run as a separate condition | - | the null control: seeds 1000-1009 |

Tier 1 = methods 0-9 (the core comparison, must all work). Tier 2 = 10-15 (group size 8 only, labelled
exploratory). Findings from reading the papers that matter for the story (details in the two spec docs):

- With 0/1 rewards several methods collapse: CoRPO at its own threshold = Dr. GRPO; RLOO = Dr. GRPO x
  G/(G-1); 2-GRPO = vanilla at G 2; LENS is mostly vanilla; MC-GRPO degenerates to +-1000 on minorities.
- The Sign paper's "DrGRPO" baseline actually divides by std, i.e. it is vanilla. Its +45-point claim
  at G = 4 (Qwen3.5-9B) is the strongest single claim this benchmark can test.
- Advantage scales differ by orders of magnitude (MC-GRPO +-1000, KRPO ~15-36, Dr. GRPO's loss / 512),
  so the equal-budget learning-rate sweep is essential, and gradient clipping will bind for some.

## 7. How to run

Laptop (Windows, CPU tests only; RTX 4060 Laptop 8 GB not used for the benchmark):
```
cd C:\GRPO
.venv\Scripts\Activate.ps1
pytest                                   # 232 tests, 1-2 minutes
```

One run (GPU):
```
python scripts/run.py --method vanilla --seed 0 --smoke                    # whole pipeline, minutes
python scripts/run.py --method vanilla --seed 0                            # one benchmark run
python scripts/run.py --method dr_grpo --seed 0 --set budget.group_size=4  # another group size
python scripts/run.py --method two_grpo --seed 0 --set budget.group_size=2
python scripts/run.py --eval-only --seed 0                                 # the untrained base model
python scripts/run.py --eval-only --val-only --seed 0                      # validation only (prompt check)
python scripts/run.py --method vanilla_null --seed 1000                    # null control
python scripts/smoke_all.py                                                # every method, one table
python scripts/make_clusters.py                                            # once, for bv_blend
python scripts/aggregate.py                                                # all results -> table + test

python scripts/tune.py plan                       # the tuning trials (8 per method), nothing runs
python scripts/tune.py run --methods dapo         # run that method's unfinished trials (validation only)
python scripts/tune.py collect                    # best trial per method -> configs/tuned.yaml (commit it)
python scripts/grid.py status                     # every block: done / stale / running / to do
python scripts/grid.py worker --block gate1       # run a block's pending jobs (HPC: hpc/grid_worker.pbs)
python scripts/figures.py                         # results/ -> results/figures/ (PDF, PNG, CSV, tables)
```
A run writes `results/<method>/g<G>/seed<k>-<time>/`: `config.yaml` (resolved), `meta.json` (seed,
git commit, dirty flag, GPU, dtype, splits), `train.jsonl` (one line per step), `eval.jsonl`,
`summary.json`, and `ckpt/`. Commit before benchmark runs: `aggregate.py` flags runs from a dirty tree
and runs that differ in GPU, dtype or commit.

Per-step diagnostics in `train.jsonl`: reward, zero_adv_frac (share of groups with all-zero advantage),
uniform_frac (all rewards equal), adv_abs_mean, reward_var, entropy, grad_norm, completion_length,
truncated_frac, clip_frac, kl, rollouts, lr; DAPO adds groups_kept, groups_dropped, refills.

Kaggle: open `notebooks/kaggle_smoke_all.ipynb`, GPU T4 x1, Internet on, secret `GITHUB_TOKEN`.

## 8. What to do next, in order

Each step has a "done when". Do not skip ahead: every later number depends on the earlier ones.

1. **GPU smoke of all methods.** Run the Kaggle notebook. Done when the table shows "yes" for all 17
   configs. Any crash: fix, add a test that reproduces it, re-run.
2. **Clusters.** The notebook's clusters cell writes `data/gsm8k_train_clusters_k16.json`. Download it,
   commit it. Done when `bv_blend` passes in the smoke table on the real data.
3. **Prompt check.** `python scripts/run.py --eval-only --val-only --no-ckpt --seed 0` with
   `--set prompt.shots=0` and with the default 3 shots (cells at the end of the Kaggle notebook).
   `--val-only` never evaluates the test split, so the choice cannot leak test information, and it
   writes to `runs/val_only/`, which `aggregate.py` never reads. Read `val_completions_step0.txt` in
   each run folder to check the format and the parser by eye. Compare validation accuracy and `val_format_rate`. Decide zero-shot vs 3-shot
   (expectation from 2610.04928: zero-shot leaves small models near 0%, so 3-shot). Record it in the
   decision log; tick the open question. Done when `prompt.shots` is final.
4. **Timing run.** One full `vanilla` run, evaluation included, ON THE GPU TYPE USED FOR EVERYTHING.
   Fill `docs/compute.md` from `python scripts/compute_report.py` (measured minutes per run -> GPU-hours
   for every block and the tuning sweep, x 2 for re-runs). Set
   `budget.rollouts_per_step` and `budget.total_rollouts` (remove PROVISIONAL). Done when the grid's
   GPU-hours are known and fit the HPC share, or the grid is cut in the plan's order (section 9).
5. **HPC access.** IIT Delhi HPC, A100-class, bf16, one GPU type for all benchmark runs. On the login
   node: `bash hpc/setup_offline.sh` (venv, pinned versions, models and GSM8K into the cache, tests;
   compute nodes have no internet). Then on a GPU node re-run `tests/test_against_trl.py` (Triton
   kernel). Done when it passes on the cluster.
6. **Gate 1.** `vanilla` at G 8, 10 seeds, against the base model (`--eval-only`: greedy decoding is
   deterministic, so one base run is enough). `qsub -P <project> -v BLOCK=gate1,MINUTES=<from compute.md>
   hpc/grid_worker.pbs`, then `python scripts/aggregate.py --results results_gate1`. Done when the gap between them is clearly bigger than
   vanilla's seed spread. If not: check the prompt and the parser, then consider 1.5B
   as the primary model. Do not run any other method before Gate 1 passes.
7. **Freeze.** After Gate 1, `configs/base.yaml` is frozen. Any later change means re-running
   everything and goes in the decision log.
8. **Author emails** for open questions (one question per email, your reading first so a yes/no
   answers it; follow up once after a week). MC-GRPO first (is +-1000 on 0/1 rewards intended?).
9. **Human re-read of Tier 2 papers** against `docs/method-specs-tier2.md` (the agents could only use a
   web-fetch tool). LENS section 5.1 first.
10. **Tuning sweep.** `scripts/tune.py` with `configs/tuning.yaml`: 8 trials per method, vanilla
    included, on VALIDATION only: lr for all (one per stratum of 1e-6 to 1e-4, log scale), plus tau
    (softmax_grpo), c (corpo), eta (ha_dw), T (bv_blend). First set the trial length in
    `configs/tuning.yaml` from the timing run. `tune.py run` per method (one HPC job each), then
    `tune.py collect` -> `configs/tuned.yaml`; commit it. If `collect` warns that a chosen lr is at the
    edge of its range (likely for KRPO and MC-GRPO, maybe Dr. GRPO and RLOO), widen that method's range
    in `lr_ranges` and re-run its whole sweep (decision log). The budget goes in the paper.
11. **Main grid.** Tier 1 at G 4, 8, 16 x 10 seeds (2-GRPO at G 2), plus `vanilla_null` (seeds
    1000-1009) and the base model: blocks `tier1` and `null_control` of `configs/grid.yaml` (281 + 10 jobs;
    G 8 comes first, so the falsification test can be read early). Several `hpc/grid_worker.pbs` jobs
    can work on one block at once. Then `scripts/aggregate.py` and `scripts/figures.py`. Check the null-control exclusion rate
    is about 5% before believing anything else.
12. **Gate 2.** Look at the results once. Apply the falsification rule exactly as written. Decide the
    paper's story. Then the 1.5B check (Tier 1, G 8), then Tier 2 (G 8) or the optional build phase.

## 9. The plan around it (summary of the Project 1 doc)

- **Research question:** under matched conditions and honest statistics, which published advantage
  estimators actually beat vanilla GRPO at small scale? Prior evidence leans null (2609.06107 for data
  policies; 2610.04928 found estimator variants within seed noise at 1B, 2 seeds). 2605.07689 claims +45
  points for Sign at G = 4. A search on 11 and 19 Oct 2026 found no paper running this comparison across
  the estimator family.
- **Protocol:** fixed model, task, prompt, rollout budget, LoRA, schedule, KL. Group size swept at 4/8/16.
  Equal tuning budget on a 500-problem validation split. Two controls: the base model and the null
  method. About 370 runs at 0.5B plus 100 at 1.5B, plus tuning and re-runs.
- **Metrics:** primary = GSM8K test exact-match, IQM over 10 seeds with stratified bootstrap 95%
  intervals. Secondary = pass@1/4/16 (RLVR can raise pass@1 while lowering pass@16). Diagnostics every
  step (zero-advantage fraction, reward variance, entropy, length, gradient norm).
- **Falsification rule:** in `README.md`. Paired bootstrap vs vanilla at G 8, 5%, Holm across the nine
  Tier 1 methods. Everything else is exploratory. Never change it after seeing results; never add
  conditions until something separates.
- **Roadmap (29 weeks, week 1 = 12 Oct 2026):** the plan put Gate 1 in week 10, Tier 1 code in weeks
  11-13, tuning in week 14, the main grid in weeks 15-19 (HPC), Gate 2 in week 20, 1.5B in 21-22, Tier 2 or
  the build phase in 23-24, diagnostics 25, related work 26, draft 27, review 28, submit 29 (early May
  2027). As of week 3 the CODE is far ahead (all 16 methods done), but nothing has run on the GPU at
  scale: the critical path is now steps 3-6 above (prompt, timing, HPC, Gate 1).
- **If behind, cut in this order:** the optional build phase, Tier 2, the 1.5B check, group sizes 4 and
  16. Never cut seeds; never skip the TRL match.
- **Paper:** 4 pages. Figure 1 accuracy per method at G 8; Figure 2 across group sizes; Figure 3
  zero-advantage fraction vs accuracy gain; Table 1 per-method numbers; Table 2 the null control.

## 10. Code still to write

| What | Status |
|---|---|
| `scripts/tune.py` | DONE 8 Nov. Stratified log-lr search, equal trials, validation only, `configs/tuned.yaml` |
| pass@k | DONE 8 Nov. `src/eval/passk.py`, in `summary.json` (`passk`), in `aggregate.py` as a secondary table |
| HPC job scripts | DONE 8 Nov. `scripts/grid.py` + `hpc/`. Resumable, claims, refuses a dirty tree or an uncommitted `tuned.yaml` |
| Figures | DONE 8 Nov. `scripts/figures.py`: figures 1-4 and tables 1-2 from `results/` with one command |
| Optional | Checkpoint/resume mid-run (advantage `state` is already plain numbers/lists/tensors so it can be saved). A run killed by the walltime is simply re-run by the grid |

Each new piece gets tests on the tiny model, like the existing ones.

## 11. Rules for whoever changes the code

1. `pytest` must pass before and after every change. Add a test for every bug you fix.
2. Nothing method-specific in `src/train/`. A method changes only its own file in `src/advantages/`
   and its own config (`advantage`, `loss`, `filter` sections).
3. Every rollout counts against the budget, including extra rollouts and dynamic-sampling refills.
4. The run seed must never change the validation split or the few-shot examples.
5. `tests/test_against_trl.py` must keep passing (on a GPU) before any benchmark run counts.
6. Never edit the falsification rule. Never pick anything using the test split.
7. Every choice that could change a number: a line in the decision log of
   `docs/implementation-notes.md`, before the run.
8. Do not quietly "fix" a method to look better: implement the paper; record deviations.

## 12. Known caveats

- Tier 2 specs were extracted by research agents through a web-fetch tool, not from downloaded PDFs.
  Every worked example matches, but a human must re-read each paper before Gate 2.
- KRPO's and MC-GRPO's advantage scales are far larger than vanilla's; their results depend heavily on
  the lr sweep covering small enough values.
- DAPO's logged training `reward` is over the KEPT (mixed) groups only. Compare methods on validation
  accuracy, not on that number.
- With one update per batch the PPO clip never acts, so NGRPO's and DAPO's clip settings have no effect
  unless `loss.num_updates` > 1. That is consistent across methods but worth stating in the paper.
- SoftmaxGRPO's tau and SignBalance's checkpoint were chosen on test sets in their papers; their
  reported gains are optimistic.
- Several papers were run at much larger scale (7B-14B, full fine-tuning). A null at 0.5B/1.5B with
  LoRA is a null at that scale only. Say so in the limitations.
