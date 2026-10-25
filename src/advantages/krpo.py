"""11. KRPO (Kalman Filter Enhanced Group Relative Policy Optimization for Language Model Reasoning,
arXiv 2505.07527, v4, section 3.2, eqs. 1-6). Code: github.com/billhhh/KRPO_LLMs_RL at 8839aaa
(train.py, group_advantages; unchanged since 9b57dfd).

What it changes: the baseline is a 1-D Kalman filter run over ONE group's rewards in row order,
reset for every group (paper section 5.1 and the code; nothing persists across steps):
    P_pred = P + Q;  K = P_pred / (P_pred + R);  x <- x + K (r_i - x);  P <- (1 - K) P_pred
    A_i = (r_i - x) / (sqrt(P) + eps)                                   eq. 6, after r_i's own update
with x0 = 0, P0 = 1 (code; the paper does not state them), Q = 1e-5, R = 1e-2, eps = 1e-8 (code).

With 0/1 rewards this is REINFORCE with a causal baseline (the mean of the EARLIER rows) and fixed
per-position weights c_i (about (i - 1) / sqrt(R i)):
- the order of a group's rows matters ([1,0,0,0] and [0,0,0,1] give different advantages);
- the first row gets about 0.0995 r: it hardly trains;
- a wrong answer after only wrong answers gets 0; an all-right group gets small positive advantages;
- the scale is large: max |A| is about 15 at G = 4, 24.5 at G = 8, 36 at G = 16, against about sqrt(G)
  for vanilla, and it grows as 1 / sqrt(R). The equal-budget lr sweep must cover this; gradient
  clipping at 1.0 will bind. Recorded in docs/implementation-notes.md.
We keep the rows in sampling order, as the code does (they are exchangeable samples).
"""
import math

import torch

from src.advantages.base import AdvantageBatch


def advantages(batch: AdvantageBatch, state: dict, Q: float = 1e-5, R: float = 1e-2, x0: float = 0.0,
               P0: float = 1.0, eps: float = 1e-8):
    r = batch.grouped().double()
    x = torch.full((r.size(0),), float(x0), dtype=torch.float64)
    P = float(P0)
    adv = torch.empty_like(r)
    for i in range(batch.group_size):                 # P and K are the same for every group
        p_pred = P + Q
        k = p_pred / (p_pred + R)
        x = x + k * (r[:, i] - x)
        P = (1 - k) * p_pred
        adv[:, i] = (r[:, i] - x) / (math.sqrt(P) + eps)
    return adv.flatten().to(torch.float32)
