"""Weeks 1-2: REINFORCE from scratch on CartPole.

Fill in every function below. Keep the signatures exactly as written,
because the tests call them this way.

    pytest            # fast checks, a few seconds
    pytest -m slow    # trains on CartPole with three seeds, a few minutes
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """Return G where G[t] = sum over k >= t of gamma**(k - t) * rewards[k].

    One backward pass over the rewards, not a double loop.
    Return a float numpy array the same length as `rewards`.
    Do not modify `rewards`.
    """
    raise NotImplementedError


class Policy(nn.Module):
    """Map a batch of observations [B, obs_dim] to action logits [B, n_actions]."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def select_action(policy: nn.Module, obs: np.ndarray) -> tuple[int, torch.Tensor]:
    """Sample one action for a single observation of shape [obs_dim].

    Return (action, log_prob):
      - action is a plain Python int
      - log_prob is a 0-dim tensor, still attached to the autograd graph
    """
    raise NotImplementedError


def reinforce_loss(
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    baseline: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar loss whose gradient is minus the REINFORCE gradient estimate.

        loss = -mean over t of  log_probs[t] * (returns[t] - baseline[t])

    `log_probs` and `returns` have shape [T]. `baseline` is None (no baseline),
    a float, or a tensor of shape [T]. No gradient may flow into `returns` or
    `baseline`. Do not normalise anything inside this function.
    """
    raise NotImplementedError


def train(
    env_id: str = "CartPole-v1",
    episodes: int = 600,
    gamma: float = 0.99,
    lr: float = 1e-2,
    seed: int = 0,
) -> list[float]:
    """Train a Policy with REINFORCE and return each episode's undiscounted return.

    Seed everything random: torch, numpy, and env.reset(seed=...) on the first episode.
    Choose your own baseline and any return normalisation, and say which in a comment.
    """
    raise NotImplementedError
