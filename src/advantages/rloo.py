"""3. RLOO (Back to Basics: Revisiting REINFORCE Style Optimization, arXiv 2402.14740, section 2.3).

What it changes: the baseline for each completion is the mean of the OTHER G - 1 completions of its
prompt (leave one out), so a completion's own reward never enters its own baseline:
    A_i = r_i - (1 / (G - 1)) sum_{j != i} r_j  =  G / (G - 1) * (r_i - mean(r)).
No division by the std.

RLOO's loss is REINFORCE on the whole sequence (sum of token log-probs, no length normalisation), so
configs/methods/rloo.yaml sets `loss.loss_type: dr_grpo` (sum over tokens / a constant). With one
update per batch the ratio is 1 and the clip never acts, which matches REINFORCE.

Expect the same results as Dr. GRPO within noise: with this loss, RLOO's advantage is Dr. GRPO's times
the constant G / (G - 1), which Adam and the learning-rate sweep absorb (method-specs.md, finding 1).
Not adopted from the paper: its KL penalty inside the reward (our beta is 0) and its 2 updates per
batch. Both recorded in docs/implementation-notes.md.
"""
from src.advantages.base import AdvantageBatch


def advantages(batch: AdvantageBatch, state: dict):
    r = batch.grouped()
    g = batch.group_size
    # Leave-one-out mean of the others: (sum of the group - own reward) / (G - 1).
    others = (r.sum(dim=1, keepdim=True) - r) / (g - 1)
    return (r - others).flatten()
