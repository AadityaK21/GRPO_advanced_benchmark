"""9. 2-GRPO (It Takes Two: Your GRPO Is Secretly DPO, arXiv 2510.00977).

What it changes: nothing in the estimator. It is vanilla GRPO at group size 2. With 0/1 rewards a
pair that disagrees gets +-0.7070 (sample std, as the verl code the authors ran; the paper's text says
+-1, the population std) and a pair that agrees gets 0. The lr sweep absorbs the scale.

The claim is about compute: at the same rollout budget, G = 2 sees 4x more distinct prompts per step
than G = 8. Our comparison is rollout-matched; the paper's was epoch-matched (256 x 2 at lr 8e-6
against 32 x 16 at lr 1e-6). State that in the paper.

The method only exists at G = 2, so GROUP_SIZE = 2: run it with --set budget.group_size=2, and the run
script refuses any other group size. Not adopted: clip-higher 0.2 / 0.28 and token-mean loss, which
are verl/DAPO defaults the authors ran, not part of the method.
"""
from src.advantages.base import AdvantageBatch
from src.advantages.grpo import group_advantages

GROUP_SIZE = 2


def advantages(batch: AdvantageBatch, state: dict):
    return group_advantages(batch.rewards, batch.group_size, scale="group")
