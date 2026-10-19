# Method specs: what each Tier 1 paper actually specifies

Extracted on 19 Oct 2026 from the papers (arXiv HTML, version noted per method) and, where it exists, the
official code at a fixed commit. Equations are the papers' own. This is the reading you implement from;
read each paper yourself before you write its file, and copy what you settle into its entry in
`implementation-notes.md`.

Every worked example uses binary rewards and group size 4 unless noted. The numbers under "Unit test" are
what our implementation must return, with our conventions (sample std, n - 1, plus 1e-4), and are
checked to 4 decimals.

---

## Findings that change the plan

1. **With 0/1 rewards several methods collapse into each other or into something degenerate.**
   - CoRPO with the paper's own threshold (0) is exactly group-mean centring: it adds nothing. It only
     differs from GRPO if you pick a threshold between 0 and 1, which the paper never did.
   - RLOO is exactly G/(G-1) times Dr. GRPO's advantage. With RLOO's own sequence-sum loss it is the same
     arm as Dr. GRPO up to a constant, which Adam and the learning-rate sweep absorb.
   - MC-GRPO's MAD (median absolute deviation) is always 0 for an odd number of 0/1 rewards. So the
     advantage is +-1/eps (+-1000 in the official code) on the minority-outcome completions and 0 on the
     rest. The paper's GSM8K evidence used a non-binary reward (2 / 1.5 / 0), where this does not happen.
   - 2-GRPO is vanilla GRPO at G = 2.
2. **The Sign-advantage paper's "DrGRPO" baseline is not Dr. GRPO.** It divides by the group std (their
   Sec. 3.1 and App. B), so it is vanilla GRPO's advantage. Their +45-point result at G = 4 is against
   that. Their base model scored 25.5% greedy but 81.4% pass@1 when sampled at temperature 1 (Table 4,
   200 questions), so much of the gain may be learning the answer format. Our 3-shot prompt should remove
   most of that; log a format rate anyway (below).
3. **Every reference implementation uses the sample std (torch.std, n - 1)**: Oat (Dr. GRPO's code),
   verl (DAPO, 2-GRPO, NGRPO), TRL. Keep n - 1 everywhere. Note what that does at G = 2: +-0.707, not
   the +-1 2-GRPO's text says.
4. **Native KL settings differ** (DeepSeekMath 0.04, MC-GRPO 0.04, SoftmaxGRPO 0.001, Dr. GRPO, DAPO and
   NGRPO 0, RLOO puts it in the reward). One shared value is required, and 0 is the native choice of most.
5. **Advantage scales differ by orders of magnitude** (MC-GRPO +-1000, SoftmaxGRPO up to G - 1, Sign +-1
   everywhere, Dr. GRPO's loss divided by 512). The equal-budget learning-rate sweep is not optional, and
   gradient-norm clipping at 1.0 will bind for some methods and not others: keep logging the gradient norm.
6. **Free parameters chosen on test sets.** SoftmaxGRPO picked tau = 0.1 as the best cell of a sweep
   reported on test accuracy. CoRPO's threshold has no binary-reward guidance. Both go into the tuning
   sweep on validation, with the same number of trials as everyone else.

## Rule for what a method's config may change

The plan: "Where a paper also changes the loss, as Dr. GRPO and DAPO do, its token aggregation and clip
range become settings in its own config. Nothing else in the training loop changes between conditions."

Applied method by method, that means **a method gets a loss or sampling change only when that change is
part of what its paper proposes**, not when it is a framework default the authors happened to run
(verl's token-mean, TRL's KL). Those are recorded as deviations instead. This keeps the comparison about
the estimator.

| # | Method | Advantage | Its own loss/sampling changes | Inherits from base |
|---|---|---|---|---|
| 0 | Vanilla GRPO | (r - mean) / (std + 1e-4) | none | everything |
| 1 | Dr. GRPO | r - mean | loss `dr_grpo` (/ 512) | clip, KL |
| 2 | DAPO | as vanilla | clip 0.2 / 0.28, loss `dapo`, dynamic sampling, overlong shaping | KL (0, as DAPO) |
| 3 | RLOO | G/(G-1) (r - mean) | loss: sequence sum, i.e. `dr_grpo` | clip, KL |
| 4 | Sign | 2r - 1 | none stated | everything |
| 5 | NGRPO | virtual reward-1 sample in mean and std | clip 0.16 / 0.24 | aggregation, KL |
| 6 | SoftmaxGRPO | G softmax(r / tau) - 1 | none (its Eq. 10 loss is VeRL's) | everything |
| 7 | CoRPO | r - max(c, mean) | none | everything |
| 8 | MC-GRPO | (r - median) / (MAD + eps) over G + 1, median dropped | 1 extra rollout per prompt | everything |
| 9 | 2-GRPO | vanilla at G = 2 | none (clip-higher and token-mean are verl/DAPO defaults) | everything |

---

## 0. Vanilla GRPO — DeepSeekMath (arXiv 2402.03300, v3, 27 Apr 2024)

- **Advantage (Sec. 4.1.2, outcome supervision):** A_i = (r_i - mean(r)) / std(r), on every token of o_i.
  The paper states neither the std convention nor an epsilon.
- **Loss (Eq. 3):** (1/G) sum_i (1/|o_i|) sum_t { min(ratio A, clip(ratio, 1 - eps, 1 + eps) A) - beta KL }:
  mean over each completion's tokens, then over the group. That is our `grpo`.
- **KL (Eq. 4):** the k3 estimator, in the loss, beta = 0.04. Clip eps: not stated.
- **Their run (Sec. 4.2):** DeepSeekMath-Instruct 7B, ~144K GSM8K/MATH questions, lr 1e-6, 64 samples per
  question, max length 1024, batch 1024, one update per batch, a learned reward model. Seeds not stated.
  GSM8K 82.9 -> 88.2.
- **Edge cases:** an all-equal group is 0/0 in the paper; with the epsilon it is 0.
- **Ours:** n - 1 and 1e-4 (as TRL), loss `grpo`, one update. **Deviation:** beta = 0 if the shared KL is 0.
- **Unit test:** [1,0,0,0] -> [1.4997, -0.4999 x3]; [1,1,1,0] -> [0.4999 x3, -1.4997]; [0,0,0,0] -> 0.

## 1. Dr. GRPO (arXiv 2503.20783, v2, 6 Oct 2025)

- **Advantage (App. A, Fig. 1):** A_i = R_i - mean(R), the std crossed out. App. A notes
  G/(G-1) x this = RLOO's advantage.
- **Loss (Sec. 3.2):** remove 1/|o_i|; "replace mask.sum(axis=dim) with a constant value (e.g.,
  generation budget)". Their listing: `(tensor*mask).sum(axis=-1)/MAX_TOKENS`, then the mean over
  completions. Ours: `dr_grpo` with the constant = `sampling.max_new_tokens` (512), never the padded width.
- **Code** (sail-sg/understand-r1-zero at dfca49d, on oat-llm 0.1.3.post1): advantage
  `rewards - rewards.view(-1, G).mean(1)`; loss `masked_sum(..., constant_normalizer=generate_max_length)`.
  Also, for both GRPO and Dr. GRPO runs: **truncated completions get reward 0** (train_zero_math.py).
- **Their settings (Table 6):** clip 0.2, beta 0, lr 1e-6 constant, 8 samples, max 3000 tokens,
  temperature 1, 3 seeds (App. C). Model: text says Qwen2.5-1.5B, scripts use Qwen2.5-Math-1.5B.
- **Unit test:** [1,0,0,0] -> [0.75, -0.25 x3]; [1,1,1,0] -> [0.25 x3, -0.75]; [0,0,0,0] -> 0.
- **Watch:** the loss is ~100x smaller than `grpo` at the same lr. The lr sweep must cover it.

## 2. DAPO (arXiv 2503.14476, v2, 20 May 2025)

DAPO does not change the advantage. It changes four things around it:
- **Clip-higher:** eps_low 0.2, eps_high 0.28 (Sec. 4.1).
- **Token-level loss (Eq. 12):** sum of token terms over the sum of lengths. Ours: `dapo` (over the
  whole batch; the paper writes it per group, the verl code per micro-batch).
- **Dynamic sampling (Sec. 3.2, Alg. 1):** drop groups whose accuracy is all 0 or all 1, keep sampling
  until the batch is full. The code filters on accuracy, not the shaped reward, and discards the excess.
  **Every generated rollout must count against our budget.** Expected rollouts per kept group, if each
  prompt has success rate p: 1 / (1 - p^G - (1 - p)^G). At G = 4: 1.14x at p = 0.5, 2.9x at p = 0.9.
- **Overlong shaping (Eq. 13):** 0 up to L_max - L_cache, then linear down to -1 at L_max, -1 beyond;
  added to the reward. Paper: L_max 20,480, L_cache 4,096 (ratio 0.2). **Ours, to settle:** L_max 512,
  L_cache 102 (same ratio), and scale the penalty by 0.5 because DAPO's reward is +-1, not 0/1 (my
  derivation, not the paper's).
- **KL:** removed (Sec. 2.3). Reward: +-1 (Eq. 7); identical to 0/1 under z-scoring, except for the
  overlong scale above.
- **Code-only settings** (verl recipe): dual-clip c = 10, weight decay 0.1, 10 warm-up steps (paper: 20).
  Not adopted; record them.
- **Needs from the harness (not built yet):** a refill loop that over-samples prompts, drops zero-variance
  groups and keeps sampling, charging every rollout to the budget.

## 3. RLOO (arXiv 2402.14740, v2, 26 Feb 2024)

- **Advantage (Sec. 2.3):** A_i = R_i - (1/(k-1)) sum_{j != i} R_j = k/(k-1) (R_i - mean R).
- **Loss:** REINFORCE on the whole sequence (sum of token log-probs, no length normalisation), no ratio,
  no clipping, KL as a sequence-level penalty inside the reward (Eq. 3). 2 gradient steps per batch.
- **Their settings:** Pythia-6.9B and Llama-7B, RLHF (TL;DR, Anthropic-HH), k = 2 and 4, lr 1e-6. No math.
- **Ours:** loss `dr_grpo` (sequence sum over a constant). One update per batch, so the ratio is 1 and
  our token-level clip never acts. beta = 0 (KL-in-reward is recorded as a deviation).
- **Unit test:** [1,0,0,0] -> [1, -1/3 x3]; [1,1,1,0] -> [1/3 x3, -1]; [0,0,0,0] -> 0.
- **Expect:** the same results as Dr. GRPO within noise. Pre-register that: it is a second check that the
  statistics behave.
- **Pitfall:** TRL's old RLOO lays the batch out as [k, B]; ours is [B, G]. Do not copy its reshape.

## 4. Sign advantage (arXiv 2605.07689, v1, 8 May 2026)

"Gradient Starvation in Binary-Reward GRPO: Why Group-Mean Centering Fails and Why the Simplest Fix Works"
(Nie et al., NTU). No code link.
- **Advantage (Sec. 3.2, Eq. 5):** A = 2r - 1: +1 correct, -1 wrong. No centring, no scaling.
- **Loss, clip, KL:** not stated (a "TRL-based" trainer). Inherit the base config.
- **Their settings:** Qwen3.5-9B, LoRA r 64, GSM8K, G = 4, lr 2e-5, 200 steps of 16 completions
  (3,200 rollouts), max completion 256, seeds 42-48, greedy evaluation on all 1,319 test problems.
- **Their results:** G = 4: Sign 73.8 (sd 8.6) vs std-normalised baseline 28.4 (sd 1.2), base 25.5,
  7 seeds, Welch p < 0.0001. G = 8: baseline 81.7 (7 seeds) vs Sign 85.8 (5 seeds). Their G = 8
  control used a 192-token cap, so G = 4 and G = 8 are not rollout-matched.
- **Edge cases:** all-correct groups get +1 everywhere, all-wrong -1 everywhere: that is the point.
- **Unit test:** [1,0,0,0] -> [1, -1 x3]; [1,1,1,0] -> [1 x3, -1]; [0,0,0,0] -> [-1 x4].
- **Watch:** with the `grpo` loss, a long wrong answer is penalised less per token. Track length.

## 5. NGRPO (arXiv 2509.18851, v1, 23 Sep 2025)

"Negative-enhanced Group Relative Policy Optimization". Code: nangongrui-ngr/NGRPO at e7a3804 (a verl fork),
`compute_ngrpo_outcome_advantage`.
- **Advantage (Sec. 4.1, Eq. 5-6):** add one virtual sample with reward r_max = 1 to **every** group, in both
  the mean and the std: A_i = (r_i - mean(R')) / (std(R') + eps), R' = R plus {1}. The virtual sample gets
  no gradient and costs no rollout.
- **Std:** Eq. 6 divides by G + 1, but the code uses torch.std (divides by G), and the paper's own Fig. 3
  numbers match the code. Use the code: n - 1 over the G + 1 values, eps 1e-6.
- **Asymmetric clipping (Sec. 4.2):** eps_neg 0.16, eps_pos 0.24, i.e. our eps_low 0.16, eps_high 0.24.
  Eq. 8 as printed is not a valid objective (its max branch gives no gradient on-policy); the code uses
  the standard clip. Use the standard clip.
- **Code-only:** an entropy bonus 0.001, KL 0, token-mean loss, and filtering of all-correct groups
  with resampling (also mentioned in App. A.1.1). Under 0/1 rewards those groups already get A = 0.
  Not adopted; record them.
- **Their run:** Qwen2.5-Math-7B, MATH, lr 1e-6, G = 8, max 3072. Seeds not stated. No GSM8K.
- **Unit test:** [1,0,0,0] -> [1.0954, -0.7303 x3]; [1,1,1,0] -> [0.4472 x3, -1.7889];
  [0,0,0,0] -> [-0.4472 x4]. Advantages in a group always sum to a negative number.

## 6. SoftmaxGRPO (arXiv 2608.09271, v1, 10 Aug 2026)

"SoftmaxGRPO: Learning to Reason using Softmax Advantage Group Estimation" (Rice). No code.
- **Advantage (Sec. 3.1, Eq. 1):** w_i = exp(R_i / tau) / sum_j exp(R_j / tau), A_i = M w_i - 1 (M = group
  size). Sums to 0; all-equal groups get 0; bounded in (-1, M - 1). Compute with a max-subtracted softmax.
  For binary rewards at small tau: correct -> M/k - 1, wrong -> about -1 (k = number correct).
- **Their GSM8K run (Table B.1):** Qwen2.5-1.5B base, tau 0.1, M = 8, lr 1e-6, clip 0.2 / 0.28, beta 1e-3,
  max 256 tokens, 1,000 iterations. GSM8K 75.8 vs GRPO 73.5. No seeds, no error bars.
- **Tau:** they recommend 0.1-0.3 for rewards in [0, 1]; 0.1 was the best cell of a sweep reported on
  test accuracy, and the sweep is not monotonic (likely noise). Ours: tune tau in {0.1, 0.3} on
  validation inside the equal tuning budget.
- **Unit test (tau 0.1):** [1,0,0,0] -> [2.9995, -0.9998 x3]; [1,1,1,0] -> [0.3333 x3, -0.9999];
  [0,0,0,0] -> 0. (tau 0.3: [2.6133, -0.8711 x3] and [0.3177 x3, -0.9530].)

## 7. CoRPO (arXiv 2511.04439, v3, 4 Mar 2026)

"CoRPO: Adding a Correctness Bias to GRPO Improves Generalization" (Cerebras). No code.
- **Advantage (Sec. 4.2, Eq. 11-12):** b = max(R_min_correct, mean(R)), A_i = R_i - b. No std in
  Algorithm 1 ("we omit the normalization term", Sec. 3.1).
- **Their task is different:** a pairwise verifier with ordinal rewards {1, 0.5, -1, -2} and
  R_min_correct = 0 (Sec. 4.4). Qwen3-8B, G = 8 or 16. No problem-solving, no GSM8K. Seeds not stated.
- **With 0/1 rewards and c = 0 it is identical to Dr. GRPO's advantage.** It only differs for
  0 < c <= 1: groups with pass rate below c get 1 - c for correct and -c for wrong (all-wrong groups get
  -c); other groups are unchanged. Sec. 4.4 says to set c near the expected reward of medium-difficulty
  prompts, which suggests about 0.5.
- **Decide before running:** c (proposal: tune in {0.25, 0.5, 0.75} on validation), and no std (as
  Algorithm 1; with std, all-wrong groups would divide by zero).
- **Unit test (c = 0.5):** [1,0,0,0] -> [0.5, -0.5 x3]; [1,1,1,0] -> [0.25 x3, -0.75];
  [0,0,0,0] -> [-0.5 x4].

## 8. MC-GRPO (arXiv 2601.22582, v1, 30 Jan 2026)

"MC-GRPO: Median-Centered Group Relative Policy Optimization for Small-Rollout Reinforcement Learning".
Code: lotusroot-kim/MC-GRPO at da07745 (TRL 0.26.1), `grpo_trainer_gsm.py`.
- **Advantage (Sec. 3, Eq. 2-6):** sample G + 1; b = median of the G + 1 rewards; MAD = median of |r - b|;
  A_i = (r_i - b) / (MAD + eps); drop the one completion whose reward is the median, train on G.
  eps is not stated in the paper; the code uses 1e-3.
- **Binary rewards:** the median is the majority label and MAD is always 0. So A = +-1000 on the
  minority completions and 0 elsewhere. In a 3-of-5-correct group the correct answers get nothing.
- **Code-only:** multiplies non-zero advantages by (G+1)/(number non-zero) (GSM trainer only); its "GRPO"
  path also samples G + 1. Not adopted.
- **Their GSM8K reward was not binary:** 2.0 exact, 1.5 numerically equivalent, 0 otherwise (App. C).
  Results (single runs): Qwen3-1.7B G = 4: 84.0 vs GRPO 81.3; G = 8: 84.6 vs 84.5. Their Table 3 says its
  numbers are placeholders: ignore it.
- **Ours:** EXTRA_ROLLOUTS = 1 (the harness already counts it: at 64 rollouts per step and G = 8,
  7 prompts x 9 = 63 rollouts), keep-mask drops the median completion, eps 1e-3, no multiplier.
- **Unit test:** [1,0,0,0,1] -> [1000, 0, 0, 0, 1000] with one 0 dropped; [1,1,1,0,0] ->
  [0, 0, 0, -1000, -1000] with one 1 dropped; [0,0,0,0,0] -> 0.

## 9. 2-GRPO (arXiv 2510.00977, v3, 14 May 2026)

"It Takes Two: Your GRPO Is Secretly DPO". No code.
- **Advantage:** GRPO at G = 2. The text (Sec. 4.3) says +1 / -1 for a disagreeing pair and 0 otherwise
  (population std); the verl code they ran uses torch.std, which gives +-0.707. Ours: vanilla at G = 2
  (+-0.7070); the lr sweep absorbs the scale.
- **Their settings:** verl, Qwen2.5-Math-1.5B and 7B, MATH and DAPO-Math, clip 0.2 / 0.28 "following
  DAPO", verl's token-mean loss, 2-GRPO at 256 prompts x 2 with lr 8e-6 vs 16-GRPO at 32 x 16 with
  lr 1e-6. No seeds, no error bars. Headline: 97.6% of 16-GRPO's score with 12.5% of the rollouts.
- **Their comparison is epoch-matched, not rollout-matched.** Ours is rollout-matched: at 64 rollouts per
  step, G = 2 sees 32 prompts per step, 4x more distinct prompts than G = 8. That is the fair test of
  their claim, and it is a confound to state in the paper.
- **Unit test:** [1,0] -> [0.7070, -0.7070]; [1,1] -> 0; [0,0] -> 0.

---

## Shared settings: evidence and recommendation

| Setting | Recommendation | Evidence |
|---|---|---|
| KL coefficient | **0** (confirmed) | TRL's default is 0 (its docs cite Dr. GRPO, DAPO, Open-Reasoner-Zero); Dr. GRPO, DAPO, NGRPO and the advantage-collapse paper (2605.21125) use 0. With beta > 0 a uniform group still gets a KL gradient (TRL issue #5588), so zero advantage would no longer mean zero gradient and the main diagnostic would blur. Deviation for vanilla (0.04), MC-GRPO (0.04), SoftmaxGRPO (0.001) |
| Token aggregation for vanilla | **`grpo`** (confirmed) | DeepSeekMath Eq. 3; the "standard" GRPO of 2610.04928 also normalises by sequence length. Framework defaults (TRL `dapo`, verl token-mean) are not vanilla GRPO. Dr. GRPO and DAPO change it as their contribution |
| Std | **n - 1, + 1e-4** (confirmed) | torch.std in Oat, verl and TRL. 2607.00152 and 2605.21125 write the population std; say which you use when comparing numbers |
| Truncated completions | **reward 0** (now set: `reward.truncated_is_wrong`) | Dr. GRPO's code does this for every arm. Otherwise a rambling answer cut off at 512 tokens scores 1 if its last number happens to match |
| Prompt | **Measure before deciding** | The plan's 3-shot rule came from 2610.04928, where a *base* model with an R1 template scored 0.08% zero-shot (18.3% at 3-shot). Ours is an *Instruct* model: verl's GSM8K example uses exactly our zero-shot instruction with Qwen2.5-0.5B-Instruct (49.6% reported; 54.3% after GRPO with LoRA). 3-shot adds about 700 prompt tokens to every training sequence, roughly doubling training cost. Run `--eval-only` with `prompt.shots=0` and `=3` on validation: use zero-shot if its format rate is at least 0.9 |
| Budget | **256 rollouts per step**, 100-200 steps (25,600-51,200), final number from the timing run | 2610.04928: 32 x 8 per step, 80 or 200 steps, 512 tokens. 2605.25252: 32 prompts per step, one epoch. Our provisional 64 per step x 200 steps (12,800) is small next to these |

## Closest papers and the scoop check (19 Oct 2026)

No one has published this benchmark. Nothing found combines many published estimators, matched rollout
and tuning budgets, several group sizes, 10 seeds, IQM with bootstrap, Holm and a null control. Arxiv
listing pages could not be browsed directly, so posts from the last few days may be missed: recheck in
week 26 as planned.

| Paper | What it does | How you differ |
|---|---|---|
| 2610.04928 (4 Oct 2026), OLMo-2 1B, GSM8K | 5 normaliser/baseline settings incl. Dr. GRPO, 2 seeds, no tests; all within seed noise (0.497-0.537). Prompt is the first-order choice | Closest. You: 15 published estimators, 10 seeds, intervals, Holm, null control, 2 scales, 3 group sizes |
| 2601.23135 v2 (1 Oct 2026), Qwen2.5-Math-1.5B LoRA | Per-prompt std vs global std vs REINFORCE, 3 seeds, all methods at REINFORCE's best lr | You give each method its own equal tuning budget |
| 2605.07689 (May 2026) | Sign vs 5 other formulations, 7 seeds, Welch t | One of your methods; its baseline is std-normalised; you test its claim at small scale |
| 2603.19335 (Mar 2026), Qwen2.5-Instruct 0.5B-7B, GSM8K | 8 algorithms (SFT, DPO family, GSPO...), 3-5 seeds, Bonferroni | Same models and task, different axis (not advantage estimators) |
| 2609.06107 DataFlex-RL (Sep 2026) | 8 data policies vs uniform, 12 seeds, paired t intervals: none excludes 0 | Your null-result precedent; different axis; means and t, not IQM and bootstrap |
| 2605.21125 (May 2026) | Defines the advantage collapse rate (share of groups with population std < 1e-6) and finds it predicts accuracy (r = -0.785) **across 245 configurations differing in model size, difficulty and G**, not across estimators or seeds | Your Figure 3 tests the claim across estimators, which they did not. Their ACR is our `uniform_frac` |
| 2607.00152 (Jun 2026) | GRPO, Dr. GRPO and DAPO as three operations on the group std; theory plus a simulation | Theory your results can confirm or refute |
| 2605.25252 (May 2026), Qwen2.5 0.5B/1.5B base, full fine-tuning | Verifier noise vs rollouts per prompt, one seed per configuration, KL 0.01, lr 5e-6. Repo results: 0.5B reaches 0.40-0.54 on 256 test questions (sampled) | A sanity range for accuracy; not a competitor |

## Diagnostics added

- **`uniform_frac`** (training): share of groups whose rewards are all equal. The same for every method,
  and equal to 2605.21125's collapse rate. `zero_adv_frac` stays method-dependent (Sign and NGRPO give
  uniform groups a signal).
- **`adv_abs_mean`** (training): the advantage scale each method trains at (finding 5).
- **Format rate** (validation and test): share of completions with an answer after "####" (finding 2).
