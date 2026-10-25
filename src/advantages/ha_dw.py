"""13. HA-DW (Your Group-Relative Advantage Is Biased, arXiv 2601.08521, v2, sections 3.1-3.2,
eqs. 7-17 and 24-25). No official code; an unmerged TRL pull request (#4872 at c9e27ec) is the only
implementation found.

What it changes: vanilla GRPO's advantage is multiplied by a weight that depends on how hard the
prompt is compared with a running "anchor" C, the model's recent success rate:
    diff = p_hat - C_prior                  p_hat = the group's success rate        eq. 13
    Phi_i = lambda * exp(-sgn(A_i) * diff)                                           eqs. 14-16
    A_i <- A_i * Phi_i                                                               eq. 17 / 25
On a prompt harder than the anchor, right answers are pushed up more and wrong answers down less;
on an easier one, the reverse. The anchor is one global number (not per prompt), updated after this
step's advantages from the batch success rate y (over ALL responses, all-equal groups included):
    C_post = (1 - eta_t) C_prior + eta_t y,   eta_t = clamp(eta * sigma_t, 0, 1)   eqs. 7-12
where sigma_t is the population std of the last m posteriors.
An all-equal group keeps advantage 0 (A = 0, so the weight does nothing).

Choices (the paper leaves them open; see docs/implementation-notes.md):
- C starts at the first batch's success rate, and the weight applies from the first step.
- sigma_t = 1 until two posteriors exist (as the TRL PR); otherwise eta_t = 0 and C never moves.
- eta = 10 (the TRL PR uses 0.1): with a small eta the anchor freezes near the first batch's accuracy,
  because eta_t = eta * sigma_t and sigma_t is the spread of the anchor itself. PROVISIONAL: compare
  0.1 and 10 on validation.
- m = 10 (TRL PR); lambda = 1.3 (the row of their Table 7 that matches their main results). Under Adam
  lambda only matters through gradient clipping.
"""
import math

import torch

from src.advantages.base import AdvantageBatch
from src.advantages.grpo import group_advantages


def advantages(batch: AdvantageBatch, state: dict, lam: float = 1.3, eta: float = 10.0, m: int = 10):
    G = batch.group_size
    base = group_advantages(batch.rewards, G, scale="group").view(-1, G)
    r = batch.grouped()
    y = r.mean().item()                                       # batch success rate, all responses
    if "C" not in state:
        state.update(C=y, hist=[], t=0)
    c_prior = state["C"]
    hist = state["hist"][-m:]
    if len(hist) >= 2:
        mean = sum(hist) / len(hist)
        sigma = math.sqrt(sum((c - mean) ** 2 for c in hist) / len(hist))
    else:
        sigma = 1.0
    eta_t = min(max(eta * sigma, 0.0), 1.0)
    diff = r.mean(dim=1, keepdim=True) - c_prior
    phi = lam * torch.exp(-torch.sign(base) * torch.sign(diff) * diff.abs())
    c_post = (1 - eta_t) * c_prior + eta_t * y                # used from the next step on
    state["C"] = c_post
    state["hist"] = (state["hist"] + [c_post])[-m:]
    state["t"] += 1
    state["eta_t"] = eta_t
    return (base * phi).flatten()
