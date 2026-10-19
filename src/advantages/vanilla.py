"""0. Vanilla GRPO (DeepSeekMath, arXiv 2402.03300, section 4.1.2, outcome supervision).

What it changes: nothing. It is the baseline every other method is compared with.
A_i = (r_i - mean(r)) / (std(r) + 1e-4), over the G completions of one prompt.

std is the sample standard deviation (divides by G - 1) and the 1e-4 keeps an all-equal group at 0,
both as TRL 1.15.0 does (the paper says only "normalized"). Recorded in docs/implementation-notes.md.
Its loss aggregation (per-completion mean, then mean over completions: the paper's 1/G sum_i 1/|o_i|
sum_t) is `loss.loss_type: grpo` in configs/base.yaml.
"""
from src.advantages.base import AdvantageBatch
from src.advantages.grpo import group_advantages


def advantages(batch: AdvantageBatch, state: dict):
    return group_advantages(batch.rewards, batch.group_size, scale="group")
