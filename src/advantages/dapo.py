"""2. DAPO (An Open-Source LLM Reinforcement Learning System at Scale, arXiv 2503.14476).

DAPO keeps vanilla GRPO's advantage. What it changes is around it, and each piece lives where it acts:
- Clip-higher, eps_low 0.2 / eps_high 0.28 (section 4.1): `loss` in configs/methods/dapo.yaml.
- Token-level loss, eq. 12 (sum of token terms over the number of tokens): `loss.loss_type: dapo`.
- Dynamic sampling, section 3.2 / Algorithm 1: `filter.drop_uniform_groups: true`. Groups whose rewards
  are all equal are dropped and fresh prompts sampled until the batch is full. Every sampled rollout
  counts against the budget (src/train/loop.py, scripts/run.py).
- Overlong reward shaping, eq. 13: here. A completion of L tokens gets a penalty added to its reward:
      0                                  if L <= L_max - L_cache
      (L_max - L_cache - L) / L_cache    if L_max - L_cache < L <= L_max   (falls linearly to -1)
      -1                                 if L > L_max
  times `overlong_scale`. Ours: L_max = 512 (sampling.max_new_tokens), L_cache = 102 (the paper's ratio,
  4,096 / 20,480 = 0.2), scale 0.5 because DAPO's reward is +-1 and ours is 0/1, so the penalty keeps the
  same size relative to the gap between right and wrong (our derivation, not the paper's; recorded in
  docs/implementation-notes.md). Dynamic sampling filters on accuracy, before this shaping, as the code does.

The shaped reward is then z-scored within the group, as vanilla (n - 1, plus 1e-4).
Not adopted (verl code only): dual-clip c = 10, weight decay 0.1, 10 warm-up steps.
"""
import torch

from src.advantages.base import AdvantageBatch
from src.advantages.grpo import group_advantages


def overlong_penalty(lengths: torch.Tensor, l_max: int, l_cache: int) -> torch.Tensor:
    """DAPO eq. 13, in [-1, 0], for completions of `lengths` tokens."""
    if not 0 < l_cache < l_max:
        raise ValueError(f"need 0 < l_cache < l_max, got l_cache={l_cache}, l_max={l_max}")
    L = lengths.to(torch.float32)
    soft = (l_max - l_cache - L) / l_cache
    return torch.where(L <= l_max - l_cache, torch.zeros_like(L), torch.clamp(soft, min=-1.0))


def advantages(batch: AdvantageBatch, state: dict, l_max: int = 512, l_cache: int = 102,
               overlong_scale: float = 0.5):
    shaped = batch.rewards + overlong_scale * overlong_penalty(batch.lengths, l_max, l_cache)
    return group_advantages(shaped, batch.group_size, scale="group")
