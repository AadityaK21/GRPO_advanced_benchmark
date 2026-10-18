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
| GPU type and precision for benchmark runs | | `configs/base.yaml` |
| Padding side; prompt tokens masked from the loss | Left padding; only completion tokens up to and including the first EOS count (`completion_mask`) | `src/train/logprobs.py` |
| Logits divided by the sampling temperature before log-probs? | Yes, before the log-softmax, for the policy, old and reference log-probs alike (as TRL) | `src/train/logprobs.py` |
| Old log-probs recomputed by the training model, detached | Once per sampled batch, before the first update, only when updates > 1; with 1 update, old = logps.detach() (as TRL) | `src/train/loop.py` |
| Updates per batch | | `configs/base.yaml` |
| Default clip range | | `configs/base.yaml` |
| KL coefficient and estimator | | `configs/base.yaml` |
| Default token aggregation | | `src/train/loss.py` |
| Group where every reward is equal: what happens | All advantages 0 (0 / (0 + 1e-4)); the group still counts in the loss denominator | `src/advantages/grpo.py` |
| Prompt format | Chat template, one user turn: the question, a space, then `Let's think step by step and output the final answer after "####".` | `scripts/train_grpo.py` |
| Advantage std: n or n−1, and the epsilon | | `src/advantages/vanilla.py` |
| GSM8K answer parser rule | | `src/reward/` |
| Validation split: which ~500 training problems, and the seed | | `configs/base.yaml` |
| How a seed sets data order and initialisation | | `src/train/loop.py` |

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
| 2026-10-19 | Full training run, our loop vs TRL 1.15.0 (`scripts/train_grpo.py` vs `scripts/train_trl.py`): Qwen2.5-0.5B-Instruct, GSM8K train, LoRA r16, lr 1e-5, 512 tokens, group 8, 1 prompt per step, seed 0, 100 steps, Kaggle T4 | Identical reward and completion length on 100 of 100 steps (same seed, same sampling stream) |
| 2026-10-19 | Control for the row above: our loop with lr 0, otherwise identical | Differs from the trained run from step 1 on (99 of 100 steps), so the match above depends on the updates, not only on sampling |
| 2026-10-19 | Peak GPU memory in that run | Ours 11.7 GB, TRL 3.2 GB: our log-probs build the full [B, C, vocab] logits; TRL chunks them |
| 2026-10-19 | `tests/test_against_trl.py` on a Kaggle T4 with TRL's Triton kernel | 53 of 53 passed |

## Decision log

Newest first. One line for every decision that could change a number.

| Date | Method | Decision | Why | Re-run needed? |
|---|---|---|---|---|
| | | | | |

## Open questions

- [ ] 

## Asking authors

1. One question per email. Cite the equation and section.
2. State your reading first, then ask whether it is right, so a yes or no is enough.
3. Follow up once after a week. If there is still no reply, implement your best reading and say so in the paper.
