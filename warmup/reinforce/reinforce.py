"""Weeks 1-2: REINFORCE from scratch on CartPole.

    pytest            # fast checks, a few seconds
    pytest -m slow    # trains on CartPole with three seeds, a few minutes
"""
from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """Return G where G[t] = sum over k >= t of gamma**(k - t) * rewards[k].

    One backward pass over the rewards, not a double loop.
    Return a float numpy array the same length as `rewards`.
    Do not modify `rewards`.
    """
    rewards = np.asarray(rewards, dtype=np.float64)   # a new array, so the caller's list is untouched
    out = np.empty_like(rewards)
    running = 0.0
    # Walk backwards: G[t] = r[t] + gamma * G[t+1]. Each step reuses the one after it, so this is O(T).
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


class Policy(nn.Module):
    """Map a batch of observations [B, obs_dim] to action logits [B, n_actions]."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        # Logits, not probabilities: Categorical(logits=...) does the softmax in a numerically stable way.
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def select_action(policy: nn.Module, obs: np.ndarray) -> tuple[int, torch.Tensor]:
    """Sample one action for a single observation of shape [obs_dim].

    Return (action, log_prob):
      - action is a plain Python int
      - log_prob is a 0-dim tensor, still attached to the autograd graph
    """
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)   # [obs_dim] -> [1, obs_dim]: the net expects a batch
    dist = torch.distributions.Categorical(logits=policy(x))
    action = dist.sample()   # sample, never argmax: the gradient estimate assumes actions drawn from the policy
    # No .detach() and no torch.no_grad(): the loss backpropagates through this log-prob into the policy.
    return int(action.item()), dist.log_prob(action).squeeze(0)


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
    returns = torch.as_tensor(returns, dtype=log_probs.dtype)
    advantage = returns if baseline is None else returns - torch.as_tensor(baseline, dtype=log_probs.dtype)
    # The advantage is a fixed weight on each log-prob, so detach it: gradient flows only through log pi(a|s).
    # The minus sign is there because optimizers minimise, and we want to maximise expected return.
    return -(log_probs * advantage.detach()).mean()


def train(
    env_id: str = "CartPole-v1",
    episodes: int = 600,
    gamma: float = 0.99,
    lr: float = 1e-2,
    seed: int = 0,
) -> list[float]:
    """Train a Policy with REINFORCE and return each episode's undiscounted return.

    Seed everything random: torch, numpy, and env.reset(seed=...) on the first episode.
    Baseline: each episode's returns are standardised (minus their mean, divided by their std).
    Subtracting the mean acts as a baseline; dividing by the std keeps the step size steady
    as episodes get longer.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_id)
    policy = Policy(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []

    obs, _ = env.reset(seed=seed)   # seed the environment once; later resets continue its random stream
    for episode in range(episodes):
        if episode > 0:
            obs, _ = env.reset()
        log_probs, rewards = [], []
        done = False
        while not done:   # one full episode with the current policy
            action, log_prob = select_action(policy, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            done = terminated or truncated

        returns = torch.as_tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = reinforce_loss(torch.stack(log_probs), returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(sum(rewards)))

    env.close()
    return history
