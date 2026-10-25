"""8. MC-GRPO (Median-Centered Group Relative Policy Optimization, arXiv 2601.22582, section 3, eqs. 2-6).

What it changes: sample one extra completion per prompt (G + 1 in all), centre on the MEDIAN reward
and scale by the median absolute deviation (MAD), then drop the one completion whose reward is the
median, so G completions enter the loss:
    b = median(r_1..r_{G+1}),  MAD = median(|r_j - b|),  A_i = (r_i - b) / (MAD + eps).
The median and MAD are robust to one outlying reward, which matters most at small G.
The extra rollout counts against the rollout budget like any other (EXTRA_ROLLOUTS below; the run
script fits fewer prompts per step to pay for it).

With 0/1 rewards the median is the majority label and MAD is always 0 (G + 1 is odd at even G). So
the minority completions get +-1/eps = +-1000 and the majority get 0: in a 3-of-5-right group the
right answers get nothing. This is what the method does here, not a bug; it is flagged as an open
question for the authors (their GSM8K reward was 2 / 1.5 / 0, where this does not happen).

Choices: eps = 1e-3 (the official code, lotusroot-kim/MC-GRPO at da07745; the paper does not state it).
The dropped completion is the FIRST one in the group whose reward equals the median. torch.median
returns the lower of the two middle values for an even count, which only happens if someone runs it
with an odd G. Not adopted (code-only): rescaling non-zero advantages by (G + 1) / (number non-zero).
"""
import torch

from src.advantages.base import AdvantageBatch

EXTRA_ROLLOUTS = 1


def advantages(batch: AdvantageBatch, state: dict, eps: float = 1e-3):
    r = batch.grouped()                                    # [P, G + 1]
    med = r.median(dim=1, keepdim=True).values
    mad = (r - med).abs().median(dim=1, keepdim=True).values
    adv = (r - med) / (mad + eps)
    # Drop one completion per group: the first whose reward equals the median.
    is_med = r == med
    first = torch.zeros_like(is_med)
    first[torch.arange(r.size(0)), is_med.float().argmax(dim=1)] = True
    keep = ~first
    adv = torch.where(keep, adv, torch.zeros_like(adv))    # the dropped one is masked out of the loss anyway
    return adv.flatten(), keep.flatten()
