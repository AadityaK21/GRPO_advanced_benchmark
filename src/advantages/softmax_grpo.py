"""6. SoftmaxGRPO (Learning to Reason using Softmax Advantage Group Estimation, arXiv 2608.09271,
section 3.1, eq. 1).

What it changes: the advantage is a softmax over the group's rewards instead of a z-score:
    w_i = exp(r_i / tau) / sum_j exp(r_j / tau),   A_i = G w_i - 1.
It sums to 0 in every group, an all-equal group gets 0, and it is bounded in (-1, G - 1). With 0/1
rewards and a small tau, a right answer in a group with k right gets about G/k - 1 and a wrong one
about -1: a rare success on a hard prompt is pushed up strongly, a failure is pushed down by about
the same amount whatever the prompt.

tau is a free parameter. The paper recommends 0.1-0.3 for rewards in [0, 1]; its 0.1 was the best
cell of a sweep reported on test accuracy. Ours: tau = 0.1 by default, tuned in {0.1, 0.3} on
validation inside the equal tuning budget (advantage.params.tau).
Its eq. 10 loss is VeRL's default, not part of the method, so the base loss is used.
"""
import torch

from src.advantages.base import AdvantageBatch


def advantages(batch: AdvantageBatch, state: dict, tau: float = 0.1):
    if tau <= 0:
        raise ValueError(f"softmax_grpo needs tau > 0, got {tau}")
    r = batch.grouped()
    w = torch.softmax(r / tau, dim=1)        # torch.softmax subtracts the max: no overflow at small tau
    return (batch.group_size * w - 1).flatten()
