"""14. SignBalance (Spurious Advantage Hidden in GRPO, arXiv 2609.04063, v1, section 3.4 step 3, eq. 11,
Appendix D, Algorithm 2). Code: "will be released"; none found on 25 Oct 2026.

What it changes: every right answer gets +c and every wrong answer gets -c * n+/n-, where n+ and n-
count the right and wrong answers IN THE GROUP. So a group's advantages sum to 0, but a right answer's
magnitude no longer depends on how many others were right:
    A+ = c,   A- = -c * n+ / n-                                            eq. 11, c = 1
With 0/1 rewards this is c (r - mean) / (1 - mean): Dr. GRPO's centring divided by the group's failure
rate. All-wrong groups get 0. No std, no epsilon.

All-right groups (n- = 0) are not covered by the paper. As Algorithm 2 is written, every row takes the
"right answer" branch and gets +c, which is what we do (`all_correct: plus_c`); `all_correct: zero` is
the other reading, for a sensitivity check. At G = 4 the model will produce many all-right groups, so
this choice matters. Differs from Sign (arXiv 2605.07689), which gives -1 to all-wrong groups; the
SignBalance paper does not cite it.
"""
import torch

from src.advantages.base import AdvantageBatch


def advantages(batch: AdvantageBatch, state: dict, c: float = 1.0, all_correct: str = "plus_c"):
    if all_correct not in ("plus_c", "zero"):
        raise ValueError(f"all_correct must be 'plus_c' or 'zero', got {all_correct!r}")
    right = batch.grouped() > 0.5
    n_pos = right.sum(dim=1, keepdim=True).float()
    n_neg = batch.group_size - n_pos
    rho = n_pos / n_neg.clamp(min=1.0)                 # the clamp only acts when n- = 0, where rho is unused
    adv = torch.where(right, torch.full_like(rho.expand_as(right), c, dtype=torch.float32), -c * rho)
    if all_correct == "zero":
        adv = torch.where(n_neg == 0, torch.zeros_like(adv), adv)
    return (adv + 0.0).flatten()
