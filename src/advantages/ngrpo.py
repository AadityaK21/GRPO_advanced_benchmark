"""5. NGRPO (Negative-enhanced Group Relative Policy Optimization, arXiv 2509.18851, section 4.1, eqs. 5-6).

What it changes: every group gets one virtual completion with the maximum reward (1). It enters the
mean and the std, but it gets no gradient and costs no rollout:
    R' = R plus {r_max},   A_i = (r_i - mean(R')) / (std(R') + eps).
An all-wrong group is no longer a zero: its completions get a negative advantage, so the model is
pushed away from them. Advantages in a group always sum to a negative number.
When it should help: when many groups are all wrong (hard prompts), which vanilla GRPO wastes.

Choices (docs/method-specs.md, section 5):
- std: the sample std over the G + 1 values (n - 1), eps 1e-6, as the official code
  (compute_ngrpo_outcome_advantage). The paper's eq. 6 divides by G + 1, but its own Fig. 3 numbers
  match the code.
- r_max = 1, the maximum of our 0/1 reward.
- Its asymmetric clip (eps_low 0.16, eps_high 0.24, section 4.2) is in configs/methods/ngrpo.yaml.
  Eq. 8 as printed has no gradient on-policy; the code uses the standard clip, and so do we.
Not adopted (code-only): an entropy bonus of 0.001, token-mean loss, and resampling of all-correct
groups. Recorded in docs/implementation-notes.md.
"""
import torch

from src.advantages.base import AdvantageBatch


def advantages(batch: AdvantageBatch, state: dict, r_max: float = 1.0, eps: float = 1e-6):
    r = batch.grouped()
    virtual = torch.full((r.size(0), 1), float(r_max), dtype=r.dtype)
    r_aug = torch.cat([r, virtual], dim=1)                 # [P, G + 1]
    mean = r_aug.mean(dim=1, keepdim=True)
    std = r_aug.std(dim=1, keepdim=True)                   # n - 1 over the G + 1 values, as torch.std
    return ((r - mean) / (std + eps)).flatten()
