"""15. LENS (Don't Waste Mistakes: Leveraging Negative RL-Groups via Confidence Reweighting,
arXiv 2510.08696, v1, section 5.1). No official code found.

What it changes: a wrong answer the model was CONFIDENT in is penalised more than a wrong answer it
was unsure of, and all-wrong groups are trained on (at weight alpha) instead of giving zero gradient.
It acts on the reward, before the usual group normalisation:

    pibar_i = exp(seq_logp_i / |o_i|)                      length-normalised probability under the
                                                           sampling policy (geometric mean per token)
    D_imp   = ( (1/G) sum_i r_i / pi_old(o_i) )^-1          eq. 14, RAW sequence probabilities
    D       = max(D_imp, 2 max_j pibar_j)  if any answer is right, else 2 max_j pibar_j
    r~_i    = r_i - (1 - r_i) (1/G) pibar_i / (D - pibar_i)          wrong answers only: in [-1/G, 0)
    A_i     = (r~_i - mean r~) / (std r~ + 1e-4)            groups with a right answer (eq. 2)
    A_i     = alpha (r~_i - mean r~)                        all-wrong groups: de-meaning only, times alpha

When it should help: when many groups are all wrong (hard prompts), which vanilla GRPO wastes.
With 0/1 rewards D_imp almost never binds (it needs whole-sequence probabilities near 1), so D is the
floor, and in a group with one wrong answer, or wrong answers of equal confidence, LENS IS vanilla GRPO.
It differs by re-ordering wrong answers by confidence, and by the small signal on all-wrong groups
(|A| < alpha / G).

Choices (docs/implementation-notes.md, LENS entry): D_imp with raw probabilities as printed, in log
space (1 / pi overflows); the floor over all G answers; the 1/G scale everywhere; sample std plus 1e-4
on mixed groups; alpha = 0.25 (the paper's, set at G = 16 and never ablated) at every G, applied to
the advantages of all-wrong groups. All-right groups get 0 under every reading.
The paper does not change the loss, and all-wrong groups must not be filtered out (no dynamic sampling).
"""
import math

import torch

from src.advantages.base import AdvantageBatch

NEEDS = ("seq_logps",)


def advantages(batch: AdvantageBatch, state: dict, alpha: float = 0.25):
    if batch.seq_logps is None:
        raise ValueError("lens needs seq_logps (the loop fills it because NEEDS asks for it)")
    G = batch.group_size
    r = batch.grouped().double()
    logp = batch.grouped(batch.seq_logps).double()                      # summed log pi_old(o | q)
    n = batch.grouped(batch.lengths).double().clamp(min=1.0)            # |o|
    pibar = torch.exp(logp / n)
    floor = 2.0 * pibar.max(dim=1, keepdim=True).values

    correct = r > 0.5
    any_correct = correct.any(dim=1, keepdim=True)
    # log D_imp = log G - logsumexp over the right answers of (-log pi); -inf where there are none.
    neg = torch.where(correct, -logp, torch.full_like(logp, -math.inf))
    log_d_imp = math.log(G) - torch.logsumexp(neg, dim=1, keepdim=True)
    d = torch.where(any_correct, torch.maximum(torch.exp(log_d_imp), floor), floor)

    r_cal = r - (1 - r) * (1.0 / G) * pibar / (d - pibar)
    centred = r_cal - r_cal.mean(dim=1, keepdim=True)
    mixed_adv = centred / (r_cal.std(dim=1, keepdim=True) + 1e-4)
    adv = torch.where(any_correct, mixed_adv, alpha * centred)
    return adv.flatten().to(torch.float32)
