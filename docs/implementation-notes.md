# Implementation notes

Every place a paper leaves a choice open, what this repo chose, and why.

Rule: if a choice could change a result, it goes in here before the run that uses it.

This file feeds the paper's implementation appendix and its limitations section. It is also your answer when a reviewer asks "is this really method X?"

---

## Status

All sixteen arXiv IDs and titles checked on 11 Oct 2026. "Official code" and "Beyond the advantage" come from each paper's abstract; confirm both when you read the paper.

| # | Method | arXiv | Tier | Official code | Beyond the advantage | Our code | Unit test | Authors contacted |
|---|---|---|---|---|---|---|---|---|
| 0 | Vanilla GRPO | 2402.03300 | 1 | none in abstract | the reference loop | | | |
| 1 | Dr. GRPO | 2503.20783 | 1 | github.com/sail-sg/understand-r1-zero | changes the loss normalisation | | | |
| 2 | DAPO | 2503.14476 | 1 | verl-based release (link in paper) | clip-higher, dynamic sampling, token-level loss, overlong reward shaping | | | |
| 3 | RLOO | 2402.14740 | 1 | none in abstract | none; proposed for RLHF and compared with PPO | | | |
| 4 | Sign advantage | 2605.07689 | 1 | none in abstract | none | | | |
| 5 | NGRPO | 2509.18851 | 1 | github.com/nangongrui-ngr/NGRPO | asymmetric clipping | | | |
| 6 | SoftmaxGRPO | 2608.09271 | 1 | none in abstract | a temperature parameter | | | |
| 7 | CoRPO | 2511.04439 | 1 | none in abstract | a correctness threshold | | | |
| 8 | MC-GRPO | 2601.22582 | 1 | github.com/lotusroot-kim/MC-GRPO | one extra rollout per prompt; count it in the budget | | | |
| 9 | 2-GRPO | 2510.00977 | 1 | none in abstract | group size 2 only | | | |
| 10 | EBPO | 2602.05165 | 2 | none in abstract | a running global prior (state) | | | |
| 11 | KRPO | 2505.07527 | 2 | github.com/billhhh/KRPOLLMsRL | a per-prompt Kalman filter (state) | | | |
| 12 | BV-Blend | 2606.28707 | 2 | none in abstract | semantic prompt clusters plus running statistics (state) | | | |
| 13 | HA-DW | 2601.08521 | 2 | none in abstract | an evolving difficulty anchor (state) | | | |
| 14 | SignBalance | 2609.04063 | 2 | "will be released" | per-class rescaling; reports parity with GRPO on open-answer math | | | |
| 15 | LENS | 2510.08696 | 2 | none in abstract | each response's likelihood | | | |

Our code: the file under `src/advantages/`. Unit test: its name in `tests/test_advantages.py`.

**What this means for the interface.** "Rewards in, advantages out" is not enough. The advantage function also needs group and prompt IDs, each response's log-probability (LENS), prompt cluster IDs (BV-Blend), and a state object that persists across steps (EBPO, KRPO, HA-DW, BV-Blend). Design the signature for all of these before you write method 1.

**Closest papers to test against or cite** (found 11 Oct 2026):

- 2605.07689: Sign advantage beats Dr. GRPO by 45 points on GSM8K at group size 4 (Qwen3.5-9B, seven seeds). The strongest claim your benchmark can confirm or refute.
- 2605.21125: names the zero-advantage fraction the "advantage collapse rate" and reports that it predicts final performance. Your Figure 3 tests this.
- 2607.00152: argues GRPO, Dr. GRPO and DAPO are three settings of the group standard deviation.
- 2605.25252: GRPO on GSM8K with Qwen2.5 0.5B and 1.5B, the same task and models as yours. Use it to sanity-check your accuracy ranges.

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
| KL coefficient and estimator | PROVISIONAL: beta = 0 (TRL's default; DAPO drops the KL term). DeepSeekMath used 0.04 with the k3 estimator. Decide before Gate 1. If not 0: k3, per token, reference = the base model (adapter off) | `configs/base.yaml` |
| Default token aggregation | `grpo`: mean over each completion's tokens, then mean over completions (DeepSeekMath eq. 3, 1/G sum_i 1/abs(o_i) sum_t). The TRL-match runs used TRL's default `dapo` instead | `configs/base.yaml`, `src/train/loss.py` |
| Group where every reward is equal: what happens | All advantages 0 (0 / (0 + 1e-4)); the group still counts in the loss denominator | `src/advantages/grpo.py` |
| Prompt format | Benchmark runs: three worked examples (fixed, from GSM8K train, calculator marks `<<...>>` removed, `#### N` kept) as earlier user and assistant turns, then the question. Each user turn is the question, a space, then `Let's think step by step and output the final answer after "####".` The TRL-match runs (`scripts/train_grpo.py`) used the last turn alone, zero-shot | `src/data.py` |
| Advantage std: n or n−1, and the epsilon | n−1 (Bessel), plus 1e-4, as TRL 1.15.0. The paper only says "normalized" | `src/advantages/vanilla.py` |
| GSM8K answer parser rule | If the text has `####`, the first number after it; otherwise the last number. `$` and thousands commas dropped, trailing decimal zeros dropped. No number = wrong. The same parser scores the reference | `src/reward/gsm8k.py` |
| Validation split: which ~500 training problems, and the seed | GSM8K train shuffled with `data.split_seed` 1234: the first 3 are the few-shot examples, the next 500 are validation, the rest (6970) train. Identical for every run; `meta.json` records the shot ids and a hash of the validation ids | `src/data.py`, `configs/base.yaml` |
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
| 2026-10-19 | all | Budget in rollouts: 64 completions per step (prompts per step = 64 // G), 12,800 per run (PROVISIONAL until the week-9 timing run) | The plan fixes rollouts, not steps, so methods that use rollouts differently get no extra compute | Not yet run |
| 2026-10-19 | all | Few-shot prompt with 3 fixed examples | The plan: 2610.04928 found zero-shot left a 1B model near 0% | Not yet run |
| 2026-10-19 | all | Default loss aggregation `grpo`, beta 0 (PROVISIONAL) | See the shared-loop rows | Not yet run |
| 2026-10-19 | all | Validation every 50 steps and before training; the best step's weights are kept and also scored on test | The plan reports final and best-on-validation accuracy | Not yet run |

## Open questions

- [ ] 

## Asking authors

1. One question per email. Cite the equation and section.
2. State your reading first, then ask whether it is right, so a yes or no is enough.
3. Follow up once after a week. If there is still no reply, implement your best reading and say so in the paper.
