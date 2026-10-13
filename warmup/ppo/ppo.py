"""Weeks 3-4: PPO from scratch on CartPole.

Fill in every function below. Keep the signatures exactly as written,
because the tests call them this way.

    pytest            # fast checks, a few seconds
    pytest -m slow    # trains on CartPole with three seeds, a few minutes
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over one rollout of length T.

    rewards[t], values[t] = V(s_t), and dones[t] = 1.0 if the episode ended
    right after step t (terminated or truncated), else 0.0.
    last_value = V(s_T), the value of the state after the final step.

        delta[t] = rewards[t] + gamma * V(s_{t+1}) * (1 - dones[t]) - values[t]
        A[t]     = delta[t] + gamma * lam * (1 - dones[t]) * A[t+1]

    Return (advantages, returns), with returns = advantages + values.
    Both are float numpy arrays of length T. One backward pass, no double loop.
    """
    T = len(rewards)
    advantages = np.zeros(T)
    next_value = last_value   # V(s_{t+1}) for the final step
    next_adv = 0.0            # A[t+1] for the final step
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + gamma * lam * (1 - dones[t]) * next_adv
        next_value, next_adv = values[t], advantages[t]
    return advantages, advantages + values


class ActorCritic(nn.Module):
    """Map observations [B, obs_dim] to (action logits [B, n_actions], values [B])."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def ppo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """PPO's clipped surrogate loss, as a scalar to minimise.

        ratio = exp(new_log_probs - old_log_probs)
        loss  = -mean( min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A) )

    All inputs have shape [N]. No gradient may flow into `old_log_probs` or
    `advantages`. Do not normalise the advantages inside this function.
    """
    raise NotImplementedError


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """0.5 * mean((values - returns) ** 2). No gradient may flow into `returns`."""
    raise NotImplementedError


def train(
    env_id: str = "CartPole-v1",
    total_steps: int = 100_000,
    rollout_len: int = 512,
    seed: int = 0,
) -> list[float]:
    """Train an ActorCritic with PPO; return the return of every finished episode, in order.

    Collect rollout_len steps, compute GAE, then run several epochs of minibatch
    updates on that rollout. Seed torch, numpy and env.reset(seed=...) once.
    Choose the remaining settings yourself (learning rate, epochs, minibatches,
    clip range, entropy and value coefficients) and say which in a comment.
    """
    raise NotImplementedError
