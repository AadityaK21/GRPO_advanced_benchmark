"""10. EBPO (Empirical Bayes Shrinkage for Stabilizing Group-Relative Policy Optimization, arXiv 2602.05165,
v3, section 3.2, eqs. 2-4, Algorithm 1). No official code found.

What it changes: each prompt's baseline is shrunk from its own group mean towards the global mean
reward, by an empirical-Bayes factor S, and the advantages are then normalised over the whole batch
(not per group):
    V_q = (1 - S) mu_group + S mu_glob                                  eq. 2
    S   = (sigma^2 / G) / (sigma^2 / G + tau^2)                          eq. 3
    A   = ((r - V_q) - mean over batch) / (std over batch + 1e-4)       eq. 4
sigma^2 is the variance of EVERY reward seen so far (pooled), tau^2 the variance of every group mean
seen so far; both are running (Welford) statistics updated with the current batch BEFORE S is computed
(Algorithm 1, Remark 3.4). There is no decay and no per-prompt state, so S is one number per step.

When it should help: at small G, where a group's own mean is noisy. Unlike vanilla, an all-wrong group
in a batch that has some right answers gets a negative advantage instead of 0.

Two consequences worth knowing (checked in tests/test_advantages.py): mu_glob cancels after the batch
centring, so the state matters only through S; and with 0/1 rewards S stays in about [1/(G+1), 1/2].
Choices: sigma^2 pooled over all rewards (Algorithm 1, not a per-prompt variance); n - 1 for every
variance and std; 1e-4; variance 0 when there are too few values; S = 0 when sigma^2 = tau^2 = 0 (then
every reward ever seen is equal and A = 0 whatever S is). Not adopted: KL 0.001 (a training default in
the paper's appendix) and the topic-ordered data stream (a sampler change, not the estimator).
"""
import torch

from src.advantages.base import AdvantageBatch


def _merge(state: dict, key: str, x: torch.Tensor) -> None:
    """Chan/Welford merge of the values x into the running (n, mean, m2) kept in state."""
    n0, mu0, m20 = state.get(key + "_n", 0), state.get(key + "_mean", 0.0), state.get(key + "_m2", 0.0)
    nb = x.numel()
    mub = x.mean().item()
    m2b = ((x - mub) ** 2).sum().item()
    n = n0 + nb
    d = mub - mu0
    state[key + "_n"] = n
    state[key + "_mean"] = mu0 + d * nb / n
    state[key + "_m2"] = m20 + m2b + d * d * n0 * nb / n


def _var(state: dict, key: str) -> float:
    n = state[key + "_n"]
    return state[key + "_m2"] / (n - 1) if n > 1 else 0.0


def advantages(batch: AdvantageBatch, state: dict, eps: float = 1e-4):
    G = batch.group_size
    r = batch.grouped().double()
    mu_group = r.mean(dim=1)
    _merge(state, "glob", r.flatten())        # every individual reward
    _merge(state, "means", mu_group)          # one mean per group
    sigma2, tau2 = _var(state, "glob"), _var(state, "means")
    den = sigma2 / G + tau2
    S = (sigma2 / G) / den if den > 0 else 0.0
    state["S_last"] = S                       # for the logs
    V = (1 - S) * mu_group + S * state["glob_mean"]
    raw = (r - V[:, None]).flatten()
    sd = raw.std() if raw.numel() > 1 else torch.zeros((), dtype=raw.dtype)
    return ((raw - raw.mean()) / (sd + eps)).to(torch.float32)
