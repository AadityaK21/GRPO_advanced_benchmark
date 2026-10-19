"""1. Dr. GRPO (Understanding R1-Zero-Like Training, arXiv 2503.20783, section 3.2).

What it changes: removes the two normalisers the paper argues bias GRPO. The advantage is not divided
by the group's standard deviation (which up-weights prompts that are nearly always right or nearly
always wrong), and the loss is not divided by each completion's own length (which lets long wrong
answers off lightly). A_i = r_i - mean(r).

The length part lives in the loss, so configs/methods/dr_grpo.yaml sets `loss.loss_type: dr_grpo`:
sum over every counted token, divided by (number of completions x max completion length).
When it should help: when length bias or difficulty bias is what holds vanilla back.
"""
from src.advantages.base import AdvantageBatch
from src.advantages.grpo import group_advantages


def advantages(batch: AdvantageBatch, state: dict):
    return group_advantages(batch.rewards, batch.group_size, scale="none")
