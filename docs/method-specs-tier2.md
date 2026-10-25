# Method specs, Tier 2: what each paper actually specifies

Extracted on 25 Oct 2026, one research agent per paper, from the arXiv HTML/PDF (version noted) and the
official code where it exists. **Read each paper yourself before a Gate-2 run**: in every case arXiv was
reached through a fetch tool, not downloaded, so equations were checked across several fetches but short
quotes are not verified against the raw text. Every worked example below was computed by the agent in
Python and then reproduced by our implementation to 4 decimals (`tests/test_advantages.py`).

Conventions as Tier 1: binary 0/1 rewards, group size 4 in the examples, sample std (n - 1), 1e-4.

---

## Findings that change the plan

1. **Two methods mostly collapse to vanilla on 0/1 rewards.**
   - LENS: its likelihood term D_imp almost never binds, and in a group with one wrong answer (or wrong
     answers of equal per-token confidence) it IS vanilla. It differs only in ordering wrong answers by
     confidence and in a small signal on all-wrong groups (|A| < alpha / G).
   - EBPO: the global mean cancels after its batch centring, so its whole running state acts through one
     scalar S per step, which stays in about [1/(G+1), 1/2] with 0/1 rewards.
2. **KRPO is REINFORCE with a causal baseline, and its scale is ~10x vanilla's.** The Kalman filter runs
   over one group in row order and is reset per group; nothing persists. Row order matters, the first
   row hardly trains, and max |A| is about 15 / 24.5 / 36 at G = 4 / 8 / 16. The lr sweep must cover it.
3. **Three methods give all-equal groups a nonzero signal**, by different routes: EBPO (shrinkage to the
   batch mean), BV-Blend (per-cluster history), LENS (all-wrong only, by confidence). SignBalance gives
   all-wrong 0 but all-right +1 (as its Algorithm 2 is written).
4. **Free parameters the papers do not give** (all recorded as decisions, several PROVISIONAL):
   HA-DW eta (and the anchor freezes if eta is small), BV-Blend's T, N0, V_prior, delta_N, encoder and K,
   SignBalance's all-correct rule.
5. **Several headline results were selected on test sets** (SignBalance: best checkpoint by the test
   average; SoftmaxGRPO in Tier 1). None of the six reports GSM8K training; SignBalance evaluates on GSM8K
   after MATH training (49.66 vs GRPO 49.89, one run).

| # | Method | Advantage, in one line | State | Needs |
|---|---|---|---|---|
| 10 | EBPO | (r - ((1-S) mu_g + S mu_glob)), batch-normalised; S from running variances | 6 running scalars | - |
| 11 | KRPO | (r_i - x_i) / sqrt(P_i): Kalman filter over the group, row order | none (reset per group) | - |
| 12 | BV-Blend | (r - b) / s, b and s blended with per-cluster history by certainty w | per-cluster moments | cluster ids |
| 13 | HA-DW | vanilla x lambda exp(-sgn(A)(p_hat - C)), C a running success-rate anchor | anchor C, last m values | - |
| 14 | SignBalance | +1 right, -n+/n- wrong (per group) | none | - |
| 15 | LENS | vanilla on a calibrated reward: wrong answers penalised by confidence; all-wrong groups x alpha | none | seq log-probs |

---

## 10. EBPO (arXiv 2602.05165, v3, 23 Feb 2026)

"EBPO: Empirical Bayes Shrinkage for Stabilizing Group-Relative Policy Optimization" (Han et al., Meta AI).
Code: none found (built on verl).
- **Advantage (Sec. 3.2, eqs. 2-4, Algorithm 1):** V_q = (1 - S) mu_group + S mu_glob;
  S = (sigma^2/G) / (sigma^2/G + tau^2); A = ((r - V_q) - mean_batch) / (std_batch + eps).
  sigma^2 = variance of every reward seen so far (pooled; Algorithm 1, "variance(W_glob)"), tau^2 =
  variance of every group mean seen so far, no -sigma^2/G correction (Remark 3.1).
- **State:** two Welford accumulators, updated with the current batch BEFORE S (Algorithm 1, Remark 3.4).
  Cumulative, no decay, no per-prompt state.
- **Hyperparameters:** none of its own. eps not stated. Training: lr 1e-6, KL 0.001 (appendix, shared by
  all their methods), G = 4 main, up to 1,000 steps, best checkpoint on validation.
- **Edge cases:** an all-equal group in a mixed batch gets -S rbar_B / std (all wrong) or +S(1 - rbar_B)/std
  (all right); a batch with one reward value gives 0; one prompt per batch gives exactly vanilla.
- **Results:** Qwen3-8B G = 4, average of 5 math benchmarks 64.39 vs GRPO 58.72 (topic-ordered data;
  shuffled "EBPO-naive" 60.00). No GSM8K.
- **Ours:** `ebpo.py`; n - 1 everywhere, 1e-4, the pure estimator under our standard sampler (EBPO-naive).
- **Unit test:** steps [1,0,0,0]+[1,1,1,0], [0,0,0,0]+[1,1,0,0], [0,0,0,0] give
  [1.3931, -0.7206 x3, 0.7206 x3, -1.3931] (S 0.3636), [-0.2397 x4, 1.5181 x2, -1.0387 x2] (S 0.3750), 0.

## 11. KRPO (arXiv 2505.07527, v4, 30 Jan 2026)

"Kalman Filter Enhanced Group Relative Policy Optimization for Language Model Reasoning" (Wang et al.).
Code: github.com/billhhh/KRPO_LLMs_RL at 8839aaa (train.py L161-200). The URL in our status table was
wrong; v4 also cites an anonymous repo that could not be opened.
- **Advantage (Sec. 3.2, eqs. 1-6):** per group, in row order: P_pred = P + Q; K = P_pred/(P_pred + R);
  x += K (r_i - x); P = (1 - K) P_pred; A_i = (r_i - x) / (sqrt(P) + eps), with the posterior x and P.
- **State:** none. "The Kalman filter state is reset for each prompt" (Sec. 5.1); the code builds a new
  filter per group. One sentence in Sec. 3.2 suggests otherwise; nothing implements it.
- **Hyperparameters:** Q 1e-5, R 1e-2 (paper = code); x0 0, P0 1, eps 1e-8 (code only).
- **0/1 rewards:** A_i = c_i (r_i - mean of rows 1..i-1, shrunk 1% to 0), c_i about (i-1)/sqrt(R i).
  Order-dependent; first row about 0.0995 r; all-right groups get small positive advantages.
- **Results:** Qwen2.5-0.5B, arithmetic: 52.904 -> 68.825; one seed; no GSM8K. Full fine-tuning, KL 0.01.
- **Ours:** `krpo.py`, code values, rows in sampling order, no rescaling (the scale difference is a
  confound for the lr sweep to absorb; recorded).
- **Unit test:** [1,0,0,0]+[1,1,1,0] -> [0.0995, -7.0481, -5.7514, -4.9725, 0.0995, 0.0705, 0.0575, -14.9376];
  [1,1,0,0] -> [0.0995, 0.0705, -11.5085, -9.9501]; [0,0,0,1] -> [0, 0, 0, 14.9873].

## 12. BV-Blend (arXiv 2606.28707, v1, 27 Jun 2026)

"BV-Blend: Uncertainty-Weighted Historical Baselines for Stable Critic-Free RL with Verifiable Rewards"
(Chang, Wu, Chang; Jilin University). Code: none found.
- **Advantage (eqs. 7-11):** v_hist = max(m2 - m1^2, 0); SEM = sqrt(v_hist)/sqrt(N_eff + delta_N);
  w = exp(-SEM/T) (0 for an unseen cluster); b = w m1 + (1 - w) mu_G; s = sqrt(w v_hist + (1 - w) sigma_G^2);
  A = (r - b)/(s + delta), delta 1e-8.
- **Clusters (eq. 5):** a frozen encoder plus a k-means codebook trained offline and fixed. Encoder, K and
  corpus are not stated in v1.
- **State (eqs. 6, 12, Algorithm 2):** per cluster m1, m2, N_eff, seen; after the advantages, clusters in
  the batch update m <- (1 - gamma) m + gamma (batch moment) with gamma = 0.9 ("update rate", so the NEW
  batch gets 0.9); new clusters initialise to N0, mean_B, mean_B^2 + V_prior. Absent clusters untouched.
- **Hyperparameters:** gamma 0.9, delta 1e-8 stated; T, N0, V_prior, delta_N not stated.
- **0/1 rewards:** v_hist tends to m1(1 - m1), so w approaches 1 for clusters near 0 or 1 success; all-right
  and all-wrong groups there get a uniform nonzero advantage; unbounded as v_hist -> 0 (no clip in paper).
- **Results:** Qwen2.5-Math-7B, math average 51.7 vs GRPO 45.5; ablation: fixed w = 0.5 scores below
  GRPO (43.7). No seeds, no GSM8K.
- **Ours:** `bv_blend.py` with T 0.1, N0 8, V_prior 0.25, delta_N 1, 1e-4, sample std for sigma_G;
  clusters from `scripts/make_clusters.py` (raw GSM8K train questions, all-MiniLM-L6-v2 at the Hub commit
  of the day, recorded; k-means K = 16, seed 0).
- **Unit test (one cluster):** step 0 [1,0,0,0]+[1,1,1,0] -> vanilla; state m1 = m2 = 0.5, N_eff 8.
  Step 1 [0,0,0,0] -> w 0.1889, A = -0.4344 x4; then m1 0.05, N_eff 4.4.

## 13. HA-DW (arXiv 2601.08521, v2, 22 Jan 2026)

"Your Group-Relative Advantage Is Biased" (Yang et al.). Code: none official; TRL PR #4872 (unmerged) at
c9e27ec, `_compute_hadw_reweighting`.
- **Advantage (eqs. 13-17, 24-25):** diff = p_hat - C_prior; Phi_i = lambda exp(-sgn(A_i) sgn(diff) |diff|);
  A_i <- vanilla A_i x Phi_i.
- **State (eqs. 7-12):** y = batch success rate over all responses; C_post = (1 - eta_t) C + eta_t y,
  eta_t = eta sigma_t, sigma_t the population std of the last m anchors; C_post is used from the next step.
- **Hyperparameters:** lambda 1.3 (their Table 7 row matching the main results); eta and m "task
  dependent", not stated (TRL PR: 0.1 and 10); initial C not stated (TRL PR: first batch's y).
- **The eta trap (agent's simulation):** with eta 0.1 or 1 the anchor freezes near the first batch's
  accuracy (sigma_t is the spread of the anchor itself); eta 10 tracks with lag. Ours: eta 10, PROVISIONAL.
- **0/1 rewards:** its native case. Group sums are no longer 0: a net push up on prompts harder than C,
  down on easier ones. lambda is a global scale, which Adam absorbs except through gradient clipping.
- **Results:** Qwen3-4B-Base average 46.5 -> 48.7; no seeds, no GSM8K.
- **Unit test (lambda 1.3, eta 0.1):** [1,0,0,0]+[1,1,1,0] -> [2.5033, -0.5061 x3, 0.5061 x3, -2.5033];
  [0,0,0,0]+[1,1,0,0] -> [0 x4, 1.1256 x2, -1.1256 x2]; C 0.475.

## 14. SignBalance (arXiv 2609.04063, v1, 3 Sep 2026)

"Spurious Advantage Hidden in GRPO" (Wang, Basu, Goswami, Yu, Tao). Code: "will be released"; none found.
- **Advantage (Sec. 3.4 step 3, eq. 11; App. D; Algorithm 2):** A+ = c, A- = -c n+/n-, counts per group, c = 1.
- **0/1 rewards:** = c (r - mean)/(1 - mean). All-wrong groups 0. All-right groups: not covered; Algorithm 2
  as written gives +c (ours, `all_correct: plus_c`); `zero` is the sensitivity check.
- **Not related to Sign (2605.07689),** which it does not cite. Sign gives all-wrong groups -1.
- **Results:** Qwen2.5-0.5B-Instruct trained on MATH, GSM8K 49.66 vs GRPO 49.89, best checkpoint chosen on
  the test average, one run. KL 1e-3.
- **Unit test:** [1,0,0,0]+[1,1,1,0]+[0,0,0,0]+[1,1,1,1] -> [1, -1/3 x3, 1 x3, -3, 0 x4, 1 x4].

## 15. LENS (arXiv 2510.08696, v1, 9 Oct 2025)

"Don't Waste Mistakes: Leveraging Negative RL-Groups via Confidence Reweighting" (Feng et al., Meta / NYU).
Code: none found. The agent could reach the paper only through a summarising fetch: re-read sec. 5.1
yourself before relying on the details below.
- **Advantage (sec. 5.1, eq. 14):** pibar = exp(seq_logp/|o|) under the sampling policy;
  D = max(D_imp, 2 max pibar) (D_imp from raw probabilities of the right answers) or 2 max pibar if none is
  right; r~ = r - (1 - r)(1/G) pibar/(D - pibar); mixed groups: vanilla on r~; all-wrong groups:
  alpha (r~ - mean). alpha 0.25 (set at G = 16, never ablated).
- **0/1 rewards:** see finding 1. Confident wrong answers are pushed down hardest.
- **Results:** Llama-3.1-8B MATH Pass@1 54.09 -> 56.63, Qwen-2.5-3B 67.06 -> 68.59 (2 runs). No GSM8K.
- **Ours:** `lens.py`, NEEDS seq_logps (our loop's summed log-probs under the sampling model), D_imp in
  log space, floor over all G, 1/G everywhere, alpha at every G.
- **Unit test:** [1,0,0,0] with log-probs [-5,-15,-25,-35], lengths [10..40] -> [1.4994, -0.5280, -0.4933,
  -0.4781]; [0,0,0,0] same -> [-0.0200, 0.0026, 0.0076, 0.0098]; equal per-token confidence -> vanilla.
