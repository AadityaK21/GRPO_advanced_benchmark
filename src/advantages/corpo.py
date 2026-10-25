"""7. CoRPO (Adding a Correctness Bias to GRPO Improves Generalization, arXiv 2511.04439,
section 4.2, eqs. 11-12, Algorithm 1).

What it changes: the baseline is never below a correctness threshold c:
    b = max(c, mean(r)),   A_i = r_i - b.
No division by the std (Algorithm 1; with the std, all-wrong groups would divide by zero).
In a group whose pass rate is below c, a wrong answer is pushed down by c and a right one up by only
1 - c: below-threshold prompts are not rewarded for being right "by luck" more than they should be.
Groups at or above c are treated exactly like Dr. GRPO.

With 0/1 rewards and the paper's own threshold (c = 0) this IS Dr. GRPO's advantage; it only differs
for 0 < c <= 1. The paper used ordinal rewards {1, 0.5, -1, -2} on a pairwise-verifier task and gives
no guidance for 0/1 rewards; section 4.4 says to set c near the expected reward of medium-difficulty
prompts. Ours: c = 0.5 by default, tuned in {0.25, 0.5, 0.75} on validation (advantage.params.c).
"""
import torch

from src.advantages.base import AdvantageBatch


def advantages(batch: AdvantageBatch, state: dict, c: float = 0.5):
    r = batch.grouped()
    baseline = torch.clamp(r.mean(dim=1, keepdim=True), min=float(c))   # max(c, mean)
    return (r - baseline).flatten()
