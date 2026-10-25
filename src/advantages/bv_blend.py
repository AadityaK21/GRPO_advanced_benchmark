"""12. BV-Blend (Uncertainty-Weighted Historical Baselines for Stable Critic-Free RL with Verifiable
Rewards, arXiv 2606.28707, v1, sections 3.2.1-3.2.4, eqs. 5-12, Algorithm 2). No official code found.

What it changes: prompts are grouped into fixed semantic clusters (offline, scripts/make_clusters.py),
and each cluster keeps running reward moments. A prompt's baseline and scale blend its own group's
mean and variance with its cluster's history, trusting the history more when it is more certain:
    v_hist = max(m2 - m1^2, 0);  SEM = sqrt(v_hist) / sqrt(N_eff + delta_N)          eqs. 7-8
    w = exp(-SEM / T)  (0 for a cluster not seen before)                               eq. 9
    b = w m1 + (1 - w) mean_G;   s = sqrt(w v_hist + (1 - w) std_G^2)                  eq. 10
    A_i = (r_i - b) / (s + 1e-4)                                                       eq. 11
After the advantages, each cluster in the batch updates from its rollouts' first and second moments:
    m1 <- (1 - gamma) m1 + gamma mean_B;  m2 likewise with mean_B(r^2);  N_eff likewise with N_B (eq. 6)
A cluster seen for the first time is initialised instead: N_eff = N0, m1 = mean_B, m2 = mean_B^2 +
V_prior (eq. 12). Clusters not in the batch are left alone. On the first step this is exactly vanilla.

With 0/1 rewards an all-right or all-wrong group in a known cluster gets a uniform, nonzero advantage
(signed by the group mean minus the cluster's history), so near-solved and hopeless prompts still give
a signal. It is unbounded as v_hist -> 0 with w -> 1 (the paper does not clip); log max |A|.

Choices (the paper gives no values for T, N0, V_prior, delta_N, the encoder or K; see
docs/implementation-notes.md): gamma = 0.9 weights the NEW batch, as eq. 6 is written ("update rate");
T = 0.1; N0 = 8 rollouts; V_prior = 0.25 (the largest 0/1 variance); delta_N = 1; sample std for
std_G; 1e-4 instead of the paper's 1e-8; a new cluster with several prompts in one batch is initialised
from all its rollouts pooled; K = 16 clusters of GSM8K train questions (all-MiniLM-L6-v2, k-means).
"""
import torch

from src.advantages.base import AdvantageBatch

NEEDS = ("cluster_ids",)


def advantages(batch: AdvantageBatch, state: dict, n_clusters: int = 16, gamma: float = 0.9, T: float = 0.1,
               N0: float = 8.0, V_prior: float = 0.25, delta_N: float = 1.0, eps: float = 1e-4):
    if batch.cluster_ids is None:
        raise ValueError("bv_blend needs cluster_ids (data.clusters_file, made by scripts/make_clusters.py)")
    cid = torch.as_tensor(batch.cluster_ids, dtype=torch.long)
    if cid.min() < 0 or cid.max() >= n_clusters:
        raise ValueError(f"cluster ids must be in [0, {n_clusters}); got {cid.min().item()}..{cid.max().item()}")
    if not state:
        state.update(m1=torch.zeros(n_clusters, dtype=torch.float64),
                     m2=torch.zeros(n_clusters, dtype=torch.float64),
                     n_eff=torch.zeros(n_clusters, dtype=torch.float64),
                     seen=torch.zeros(n_clusters, dtype=torch.bool))
    m1, m2, n_eff, seen = state["m1"], state["m2"], state["n_eff"], state["seen"]
    G = batch.group_size
    r = batch.grouped().double()

    # The advantage, from the state BEFORE this batch (Algorithm 2).
    mu_g, var_g = r.mean(dim=1), r.var(dim=1)                       # sample variance, as torch.std
    v_hist = (m2[cid] - m1[cid] ** 2).clamp(min=0.0)
    sem = v_hist.sqrt() / (n_eff[cid] + delta_N).sqrt()
    w = torch.where(seen[cid], torch.exp(-sem / T), torch.zeros_like(sem))
    b = w * m1[cid] + (1 - w) * mu_g
    s = (w * v_hist + (1 - w) * var_g).sqrt()
    adv = (r - b[:, None]) / (s[:, None] + eps)
    state["w_mean"] = w.mean().item()                                 # for the logs

    # Then the update, from this batch's rollouts per cluster (eqs. 6 and 12).
    per_row = cid.repeat_interleave(G)
    flat = r.flatten()
    n_b = torch.zeros(n_clusters, dtype=torch.float64).index_add_(0, per_row, torch.ones_like(flat))
    s1 = torch.zeros(n_clusters, dtype=torch.float64).index_add_(0, per_row, flat)
    s2 = torch.zeros(n_clusters, dtype=torch.float64).index_add_(0, per_row, flat * flat)
    present = n_b > 0
    mean_b, mean2_b = s1 / n_b.clamp(min=1), s2 / n_b.clamp(min=1)
    old = present & seen
    m1[old] = (1 - gamma) * m1[old] + gamma * mean_b[old]
    m2[old] = (1 - gamma) * m2[old] + gamma * mean2_b[old]
    n_eff[old] = (1 - gamma) * n_eff[old] + gamma * n_b[old]
    new = present & ~seen
    n_eff[new] = N0
    m1[new] = mean_b[new]
    m2[new] = mean_b[new] ** 2 + V_prior
    seen[new] = True
    return adv.flatten().to(torch.float32)
