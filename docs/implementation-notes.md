# Implementation notes

Every place a paper leaves a choice open, what this repo chose, and why.

Rule: if a choice could change a result, it goes in here before the run that uses it.

This file feeds the paper's implementation appendix and its limitations section. It is also your answer when a reviewer asks "is this really method X?"

---

## Status

All sixteen arXiv IDs and titles checked on 11 Oct 2026. "Official code" and "Beyond the advantage" come from each paper's abstract; confirm both when you read the paper.

| # | Method | arXiv | Tier | Official code | Beyond the advantage | Our code | Unit test | Authors contacted |
|---|---|---|---|---|---|---|---|---|
| 0 | Vanilla GRPO | 2402.03300 | 1 | none in abstract | the reference loop | `vanilla.py` | `test_vanilla_spec_example` | |
| 1 | Dr. GRPO | 2503.20783 | 1 | github.com/sail-sg/understand-r1-zero | changes the loss normalisation | `dr_grpo.py` | `test_dr_grpo_hand_worked_example` | |
| 2 | DAPO | 2503.14476 | 1 | verl-based release (link in paper) | clip-higher, dynamic sampling, token-level loss, overlong reward shaping | `dapo.py` + `filter` | `test_dapo_*`, `test_dynamic_sampling_*` | |
| 3 | RLOO | 2402.14740 | 1 | none in abstract | none; proposed for RLHF and compared with PPO | `rloo.py` | `test_rloo_is_leave_one_out` | |
| 4 | Sign advantage | 2605.07689 | 1 | none in abstract | none | `sign.py` | `test_sign_ignores_the_group` | |
| 5 | NGRPO | 2509.18851 | 1 | github.com/nangongrui-ngr/NGRPO | asymmetric clipping | `ngrpo.py` | `test_ngrpo_adds_a_virtual_correct_sample` | |
| 6 | SoftmaxGRPO | 2608.09271 | 1 | none in abstract | a temperature parameter | `softmax_grpo.py` | `test_softmax_grpo` | |
| 7 | CoRPO | 2511.04439 | 1 | none in abstract | a correctness threshold | `corpo.py` | `test_corpo_threshold` | |
| 8 | MC-GRPO | 2601.22582 | 1 | github.com/lotusroot-kim/MC-GRPO | one extra rollout per prompt; count it in the budget | `mc_grpo.py` | `test_mc_grpo_*` | |
| 9 | 2-GRPO | 2510.00977 | 1 | none in abstract | group size 2 only | `two_grpo.py` | `test_two_grpo_is_vanilla_at_group_size_2_only` | |
| 10 | EBPO | 2602.05165 | 2 | none found | a running global prior (state) | `ebpo.py` | `test_ebpo_*` | |
| 11 | KRPO | 2505.07527 | 2 | github.com/billhhh/KRPO_LLMs_RL @ 8839aaa | a Kalman filter over each group, reset per group (no state) | `krpo.py` | `test_krpo_spec_example` | |
| 12 | BV-Blend | 2606.28707 | 2 | none found | semantic prompt clusters plus running statistics (state) | `bv_blend.py` + `scripts/make_clusters.py` | `test_bv_blend_*` | |
| 13 | HA-DW | 2601.08521 | 2 | none official (TRL PR #4872, unmerged) | an evolving difficulty anchor (state) | `ha_dw.py` | `test_ha_dw_spec_example` | |
| 14 | SignBalance | 2609.04063 | 2 | "will be released"; none found 25 Oct 2026 | per-class rescaling; reports parity with GRPO on open-answer math | `signbalance.py` | `test_signbalance_spec_example` | |
| 15 | LENS | 2510.08696 | 2 | none found | each response's likelihood | `lens.py` | `test_lens_*` | |

Our code: the file under `src/advantages/`. Unit test: its name in `tests/test_advantages.py`.

**What this means for the interface.** "Rewards in, advantages out" is not enough. The advantage function also needs group and prompt IDs, each response's log-probability (LENS), prompt cluster IDs (BV-Blend), and a state object that persists across steps (EBPO, KRPO, HA-DW, BV-Blend). Design the signature for all of these before you write method 1.

**Closest papers to test against or cite** (found 11 Oct 2026):

- 2605.07689: Sign advantage beats Dr. GRPO by 45 points on GSM8K at group size 4 (Qwen3.5-9B, seven seeds). The strongest claim your benchmark can confirm or refute.
- 2605.21125: names the zero-advantage fraction the "advantage collapse rate" and reports that it predicts final performance. Your Figure 3 tests this.
- 2607.00152: argues GRPO, Dr. GRPO and DAPO are three settings of the group standard deviation.
- 2605.25252: GRPO on GSM8K with Qwen2.5 0.5B and 1.5B (base models, full fine-tuning, one seed per configuration). Use it to sanity-check your accuracy ranges.
- Found on 19 Oct 2026 (scoop check, none found): 2610.04928 (closest: 5 estimator settings on OLMo-2 1B, GSM8K, 2 seeds), 2601.23135 v2, 2603.19335, 2609.06107. Details and how you differ: `docs/method-specs.md`.

---

## Shared loop

These settings apply to every method unless the method's entry overrides them. Fill them in once, before Gate 1. Changing one later means re-running everything.

| Setting | Value | Where it lives |
|---|---|---|
| TRL version (reference only) | 1.15.0 | `pyproject.toml` |
| torch and transformers versions | torch 2.14.1, transformers 5.19.0 (datasets 5.1.0, Python 3.14 on the laptop) | `pyproject.toml` |
| GPU type and precision for benchmark runs | Planned: one A100-class type, bf16, for every benchmark run (`model.dtype: auto` picks bf16 there). Kaggle T4 runs are fp32 and are for development only. Each run records its GPU and dtype; `aggregate.py` warns if compared runs differ | `configs/base.yaml` |
| Padding side; prompt tokens masked from the loss | Left padding; only completion tokens up to and including the first EOS count (`completion_mask`) | `src/train/logprobs.py` |
| Logits divided by the sampling temperature before log-probs? | Yes, before the log-softmax, for the policy, old and reference log-probs alike (as TRL) | `src/train/logprobs.py` |
| Old log-probs recomputed by the training model, detached | Once per sampled batch, before the first update, only when updates > 1; with 1 update, old = logps.detach() (as TRL) | `src/train/loop.py` |
| Updates per batch | 1 (TRL's default). With 1, the ratio is exactly 1 and clipping never acts | `configs/base.yaml` |
| Default clip range | eps_low = eps_high = 0.2 (PPO's and TRL's default). DAPO changes it in its own config | `configs/base.yaml` |
| KL coefficient and estimator | beta = 0. Evidence (method-specs.md): TRL's default, Dr. GRPO, DAPO, NGRPO all use 0, and with beta > 0 uniform groups still get a KL gradient, so zero advantage would no longer mean zero gradient. DeepSeekMath used 0.04 (k3): a recorded deviation for vanilla, MC-GRPO and SoftmaxGRPO. Confirm before Gate 1 | `configs/base.yaml` |
| Truncated completions | Reward 0 (`reward.truncated_is_wrong`), as Dr. GRPO's code does for every arm. Confirm before Gate 1 | `src/train/loop.py` |
| Default token aggregation | `grpo`: mean over each completion's tokens, then mean over completions (DeepSeekMath eq. 3, 1/G sum_i 1/abs(o_i) sum_t). The TRL-match runs used TRL's default `dapo` instead | `configs/base.yaml`, `src/train/loss.py` |
| Group where every reward is equal: what happens | All advantages 0 (0 / (0 + 1e-4)); the group still counts in the loss denominator | `src/advantages/grpo.py` |
| Prompt format | Benchmark runs: three worked examples (fixed, from GSM8K train, calculator marks `<<...>>` removed, `#### N` kept) as earlier user and assistant turns, then the question. Each user turn is the question, a space, then `Let's think step by step and output the final answer after "####".` The TRL-match runs (`scripts/train_grpo.py`) used the last turn alone, zero-shot | `src/data.py` |
| Advantage std: n or n−1, and the epsilon | n−1 (Bessel), plus 1e-4, as TRL 1.15.0. The paper only says "normalized" | `src/advantages/vanilla.py` |
| GSM8K answer parser rule | If the text has `####`, the first number after it; otherwise the last number. `$` and thousands commas dropped, trailing decimal zeros dropped. No number = wrong. The same parser scores the reference | `src/reward/gsm8k.py` |
| Validation split: which ~500 training problems, and the seed | GSM8K train shuffled with `data.split_seed` 1234: the first 500 are validation, the next 3 the few-shot examples, the rest (6970) train. The validation set does not depend on the number of shots. Identical for every run; `meta.json` records the shot ids and a hash of the validation ids | `src/data.py`, `configs/base.yaml` |
| How a seed sets data order and initialisation | The run seed sets `random` and `torch` before the model loads: it fixes the training-pool order, LoRA's initialisation and the sampling stream. It does not touch the splits or the few-shot examples. Same seed, same machine: identical run | `scripts/run.py` |

---

## Method entries

Copy this block for each method when you start it.

### `#` Method (arXiv id)

- **Paper version read:** v_, dated ____
- **Official code:** link at a fixed commit, or "none found"
- **What it changes (one sentence):**
- **When it should help (one sentence):**
- **Implemented from:** eq. __, section __, in `src/advantages/____.py`
- **Config keys it owns:**
- **Unit test:** `test_____`. Hand-worked example: rewards ____ give advantages ____

**Ambiguities**

| # | Question | Options | Choice | Why | Evidence |
|---|---|---|---|---|---|
| 1 | | | | | paper section, code line, or author reply |

**Deviations from the paper:** what differs from the paper's setup and why (budget matching, group size, model).

**Author correspondence:** date sent, who, the question in one line, reply date, the answer quoted, resolved or not. Keep email addresses out of the repo.

**Check against the paper's own numbers:** their setting, ours, what we saw, verdict.

---

### Example: 0 Vanilla GRPO (arXiv 2402.03300)

Settle these from the paper before writing code, then fill in Choice, Why and Evidence.

| # | Question | Options | Choice | Why | Evidence |
|---|---|---|---|---|---|
| 1 | Is each response's loss divided by its length? | per-response mean / token mean over the batch / a constant | | | |
| 2 | Which std goes in the advantage, and is an epsilon added? | n or n−1; epsilon value | | | |
| 3 | What happens when every reward in a group is equal? | advantage 0 / drop the group | | | |
| 4 | Which KL estimator and coefficient, and is it applied per token? | | | | |
| 5 | How many updates per batch, and what clip range? | | | | |

---

## Verification log

Evidence that the shared loop is a correct GRPO. Newest first.

| Date | Check | Result |
|---|---|---|
| 2026-10-19 | Chunked log-probs on the GPU: our loop vs TRL 1.15.0, same setting as the 100-step match below, 20 steps, Kaggle T4 | Identical reward and completion length on 20 of 20 steps. Peak GPU memory 3.19 GB (was 11.7 GB; TRL 3.2 GB) |
| 2026-10-19 | `scripts/run.py --smoke` on a Kaggle T4: 3-shot prompt, 2 steps of 16 completions, validation and test on 16 problems each | Runs end to end in 1.6 min, peak 3.1 GB, writes config, meta, train/eval logs and summary |
| 2026-10-19 | Chunked log-probs (`src/train/logprobs.py`, 256 tokens per piece) against all-at-once, `tests/test_logprobs.py` | Same log-probs and gradients (1e-5), also with gradient checkpointing and with LoRA's adapter-off reference. CPU memory for 8 × 512 tokens at a 50k vocabulary: 3.9 GB → 0.35 GB |
| 2026-10-19 | Micro-batches against the full batch, every loss type, `tests/test_loop.py` | Same update (1e-4) |
| 2026-10-19 | Statistics under the null, simulated (`tests/test_null.py`) | Unpaired IQM intervals at 10 vs 10 runs exclude 0 in 4.6% of null splits. The paired test rejects 6.0–6.5% under the null: slightly above 5%, so the real null control matters |
| 2026-10-19 | Full training run, our loop vs TRL 1.15.0 (`scripts/train_grpo.py` vs `scripts/train_trl.py`): Qwen2.5-0.5B-Instruct, GSM8K train, LoRA r16, lr 1e-5, 512 tokens, group 8, 1 prompt per step, seed 0, 100 steps, Kaggle T4 | Identical reward and completion length on 100 of 100 steps (same seed, same sampling stream) |
| 2026-10-19 | Control for the row above: our loop with lr 0, otherwise identical | Differs from the trained run from step 1 on (99 of 100 steps), so the match above depends on the updates, not only on sampling |
| 2026-10-19 | Peak GPU memory in that run | Ours 11.7 GB, TRL 3.2 GB: our log-probs build the full [B, C, vocab] logits; TRL chunks them |
| 2026-10-19 | `tests/test_against_trl.py` on a Kaggle T4 with TRL's Triton kernel | 53 of 53 passed |

## Decision log

Newest first. One line for every decision that could change a number.

| Date | Method | Decision | Why | Re-run needed? |
|---|---|---|---|---|
| 2026-10-25 | Tier 2 | Implemented from `docs/method-specs-tier2.md` (one agent per paper; every worked example reproduced to 4 decimals). Re-read each paper before Gate 2: arXiv was reached only through a fetch tool | Specs need a human check before they cost GPU time | Not yet run |
| 2026-10-25 | EBPO | sigma^2 pooled over all rewards (Algorithm 1), n - 1 everywhere, 1e-4, stats updated before S, no decay; the plain estimator (EBPO-naive), not the topic-ordered stream | Algorithm 1; the topic ordering is a sampler change | Not yet run |
| 2026-10-25 | KRPO | Filter per group, reset each group, rows in sampling order; code's x0 0, P0 1, Q 1e-5, R 1e-2, eps 1e-8; no rescaling | Sec. 5.1 and the code. Its advantages are ~10x vanilla's: the lr sweep and the shared clip must absorb it | Not yet run |
| 2026-10-25 | BV-Blend | gamma 0.9 on the NEW batch (eq. 6 as written), T 0.1, N0 8, V_prior 0.25, delta_N 1, 1e-4, sample std; clusters: raw GSM8K train questions, all-MiniLM-L6-v2 (Hub commit recorded in the file), k-means K 16 seed 0; `run.py` refuses a clusters file made from other questions | Paper gives only gamma and delta; the rest are our choices, recorded | Not yet run |
| 2026-10-25 | HA-DW | lambda 1.3, m 10, C starts at the first batch's accuracy and the weight applies from step 1, sigma 1 until two anchors exist, eta_t clamped to [0, 1]; eta 10 PROVISIONAL | With eta 0.1 (TRL PR) the anchor freezes near the first batch's accuracy (agent's simulation) | Not yet run |
| 2026-10-25 | SignBalance | c 1, counts per group, all-right groups +1 (Algorithm 2 as written), no epsilon | `all_correct: zero` kept as the sensitivity check | Not yet run |
| 2026-10-25 | LENS | Sec. 5.1 version: D_imp from raw probabilities in log space, floor over all G, 1/G on every wrong answer, mixed groups vanilla on r~, all-wrong groups alpha x de-meaned, alpha 0.25 at every G | As printed; alpha was set at G = 16 and never ablated | Not yet run |
| 2026-10-25 | all | A completion with no counted tokens (a keep-mask row) is out of the loss entirely, including the per-completion denominators of `grpo` and `dr_grpo` | MC-GRPO trains on G of its G + 1 completions; without this, the dropped row would shrink every other row's weight by G/(G+1). No effect without a keep-mask (every sampled row has at least one counted token), so earlier results and the TRL match stand | No |
| 2026-10-25 | DAPO | Dynamic sampling refills only the missing groups (not a whole new batch as verl does), stops after `filter.max_refills` = 10 rounds or when the next round would pass the budget, and trains on whatever mixed groups it has; zero mixed groups = no update, rollouts still counted | Every rollout counts against the same budget as every other method; sampling only the deficit wastes none. The run ends when the budget is spent, and the lr decays with rollouts spent, not steps | Not yet run |
| 2026-10-25 | DAPO | Overlong shaping at L_max 512, L_cache 102, penalty x0.5; filtered on accuracy before shaping | Same L_cache / L_max ratio as the paper (0.2); x0.5 because DAPO's reward is +-1 and ours 0/1 (our derivation); the verl code filters on accuracy | Not yet run |
| 2026-10-25 | MC-GRPO | eps 1e-3 (code), the FIRST completion equal to the median is the one dropped, no (G+1)/nonzero rescale | Paper silent on eps and on which tied completion drops; the rescale is in the GSM trainer only | Not yet run |
| 2026-10-25 | NGRPO | Sample std (n - 1) over the G + 1 values, eps 1e-6, standard clip with eps_low 0.16 / eps_high 0.24; no entropy bonus, no all-correct resampling | The official code; the paper's eq. 6 (divide by G + 1) disagrees with its own Fig. 3, the code agrees | Not yet run |
| 2026-10-25 | Sign | Rewards outside [0, 1] raise | A = 2r - 1 is defined for 0/1 rewards only | No |
| 2026-10-25 | 2-GRPO | Its own file (`GROUP_SIZE = 2`); `run.py` refuses any other `budget.group_size` | A method config cannot change the shared budget, so the guard lives in the method | No |
| 2026-10-25 | RLOO, SoftmaxGRPO, CoRPO | RLOO uses loss `dr_grpo`; SoftmaxGRPO tau 0.1 and CoRPO c 0.5 by default, both tuned on validation | method-specs.md, sections 3, 6, 7 | Not yet run |
| 2026-10-19 | all | A method's config changes the loss or sampling only where its paper proposes that change; framework defaults its authors ran (verl token-mean, TRL KL) are recorded as deviations | Keeps the comparison about the estimator, as the plan requires | Not yet run |
| 2026-10-19 | all | Truncated completions score 0; log `uniform_frac`, `adv_abs_mean`, and the format rate | Dr. GRPO's code; method-specs.md findings 2 and 5 | Not yet run |
| 2026-10-19 | all | Budget in rollouts: 64 completions per step (prompts per step = 64 // G), 12,800 per run (PROVISIONAL until the week-9 timing run) | The plan fixes rollouts, not steps, so methods that use rollouts differently get no extra compute | Not yet run |
| 2026-10-19 | all | Few-shot prompt with 3 fixed examples | The plan: 2610.04928 found zero-shot left a 1B model near 0% | Not yet run |
| 2026-10-19 | all | Default loss aggregation `grpo`, beta 0 (PROVISIONAL) | See the shared-loop rows | Not yet run |
| 2026-10-19 | all | Validation every 50 steps and before training; the best step's weights are kept and also scored on test | The plan reports final and best-on-validation accuracy | Not yet run |

## Open questions

- [ ] Prompt: zero-shot or 3-shot? Measure base validation accuracy and format rate both ways (`--eval-only`), then decide. See method-specs.md
- [ ] Budget: rollouts per step (proposal 256) and total, from the timing run
- [ ] CoRPO's threshold c under 0/1 rewards (paper used 0, which equals Dr. GRPO's advantage). Proposal: tune c in {0.25, 0.5, 0.75} on validation
- [ ] MC-GRPO under 0/1 rewards: MAD is always 0, so A = +-1/eps on minority completions only. Email the author: is that intended, and what eps? (Paper silent; code 1e-3)
- [ ] SoftmaxGRPO's tau: tune in {0.1, 0.3} on validation (paper's 0.1 was picked on test accuracy)
- [x] DAPO: refill loop for dynamic sampling built (every rollout counted, budget never passed) and overlong shaping at 512 tokens (L_cache 102, penalty x0.5). 25 Oct 2026
- [ ] HA-DW's eta: 10 (ours) or 0.1 (TRL PR, where the anchor freezes)? Compare on validation inside the tuning budget
- [ ] BV-Blend's T (0.1 ours; paper silent): include {0.05, 0.1, 0.2} in its tuning trials. Log mean w and max |A| per step
- [ ] SignBalance on all-right groups: +1 (ours, Algorithm 2) vs 0. Run `all_correct=zero` as a sensitivity check at G = 4
- [ ] KRPO's scale (max |A| ~15-36): check its lr sweep covers 10x smaller lrs than vanilla's
- [ ] LENS: re-read sec. 5.1 from the PDF (the agent only had a summarising fetch) before Gate 2
- [ ] Run `scripts/make_clusters.py` once (Kaggle), commit `data/gsm8k_train_clusters_k16.json`
- [ ] DAPO: the training `reward` it logs is over the kept (mixed) groups only. Compare its training curve with others on validation accuracy, not on that number
- [ ] 

## Asking authors

1. One question per email. Cite the equation and section.
2. State your reading first, then ask whether it is right, so a yes or no is enough.
3. Follow up once after a week. If there is still no reply, implement your best reading and say so in the paper.
