"""4. Sign advantage (Gradient Starvation in Binary-Reward GRPO, arXiv 2605.07689, section 3.2, eq. 5).

What it changes: no centring and no scaling at all. A_i = 2 r_i - 1: +1 for a right answer, -1 for a
wrong one, whatever the rest of the group did.
When it should help: at small group sizes, where many groups are all right or all wrong and get a zero
advantage under group-mean centring ("gradient starvation"). Here those groups still push: all-wrong
groups are pushed down everywhere, all-right groups up.

Rewards are assumed to be 0/1 (exact match). A reward strictly between 0 and 1 is mapped linearly,
2r - 1, which is the paper's formula; it is not defined for other reward scales, so those raise.
Loss, clip and KL: the paper does not state them (a TRL-based trainer), so the base config is used.
Watch: under the `grpo` loss a long wrong answer is pushed down less per token. Track length.
"""
import torch

from src.advantages.base import AdvantageBatch


def advantages(batch: AdvantageBatch, state: dict):
    r = batch.rewards
    if (r < 0).any() or (r > 1).any():
        raise ValueError("sign advantage needs rewards in [0, 1]: A = 2r - 1 (arXiv 2605.07689, eq. 5)")
    return (2 * r - 1).to(torch.float32)
