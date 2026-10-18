"""Weeks 3-4: PPO from scratch on CartPole.

Fill in every function below. Keep the signatures exactly as written,
because the tests call them this way.

    pytest            # fast checks, a few seconds
    pytest -m slow    # trains on CartPole with three seeds, a few minutes
"""
from __future__ import annotations

import gymnasium as gym
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
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_actions))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)


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
    advantages = advantages.detach()
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """0.5 * mean((values - returns) ** 2). No gradient may flow into `returns`."""
    return 0.5 * ((values - returns.detach()) ** 2).mean()


def train(
    env_id: str = "CartPole-v1",
    total_steps: int = 100_000,
    rollout_len: int = 512,
    seed: int = 0,
) -> list[float]:
    """Train an ActorCritic with PPO; return the return of every finished episode, in order.

    Settings, following the "37 implementation details" post: Adam with lr 2.5e-4 (eps 1e-5)
    decayed linearly to 0; gamma 0.99, lambda 0.95; clip 0.2; 4 epochs of 4 minibatches per
    rollout; value coefficient 0.5; entropy coefficient 0.01; gradient norm clipped at 0.5;
    advantages standardised within each minibatch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_id)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    model = ActorCritic(obs_dim, n_actions)
    lr = 2.5e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    gamma, lam, clip_eps, epochs, n_minibatches, ent_coef, vf_coef = 0.99, 0.95, 0.2, 4, 4, 0.01, 0.5

    obs, _ = env.reset(seed=seed)
    episode_return, history = 0.0, []
    n_updates = max(1, total_steps // rollout_len)
    for update in range(n_updates):
        optimizer.param_groups[0]["lr"] = (1.0 - update / n_updates) * lr   # linear decay to 0

        # 1. Collect one rollout with the current policy. Nothing here needs gradients.
        obs_buf = np.zeros((rollout_len, obs_dim), np.float32)
        act_buf = np.zeros(rollout_len, np.int64)
        logp_buf = np.zeros(rollout_len, np.float32)
        rew_buf = np.zeros(rollout_len, np.float32)
        done_buf = np.zeros(rollout_len, np.float32)
        val_buf = np.zeros(rollout_len, np.float32)
        for t in range(rollout_len):
            obs_buf[t] = obs
            with torch.no_grad():
                logits, value = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            act_buf[t], logp_buf[t], val_buf[t] = action.item(), dist.log_prob(action).item(), value.item()
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            rew_buf[t], done_buf[t] = reward, float(terminated or truncated)
            episode_return += reward
            if terminated or truncated:
                history.append(float(episode_return))
                episode_return = 0.0
                obs, _ = env.reset()

        # 2. Advantages and returns from the rollout. last_value bootstraps the unfinished episode.
        with torch.no_grad():
            last_value = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[1].item()
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_value, gamma, lam)
        b_obs, b_act = torch.as_tensor(obs_buf), torch.as_tensor(act_buf)
        b_old_logp = torch.as_tensor(logp_buf)   # the policy that collected the data: the ratio's denominator
        b_adv = torch.as_tensor(advantages, dtype=torch.float32)
        b_ret = torch.as_tensor(returns, dtype=torch.float32)

        # 3. Several epochs of minibatch updates on the same rollout. The clipped ratio keeps the
        #    policy close to the one that collected the data, which is what makes reuse safe.
        idx = np.arange(rollout_len)
        minibatch = rollout_len // n_minibatches
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, rollout_len, minibatch):
                j = idx[start:start + minibatch]
                logits, values = model(b_obs[j])
                dist = torch.distributions.Categorical(logits=logits)
                mb_adv = b_adv[j]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                loss = (
                    ppo_policy_loss(dist.log_prob(b_act[j]), b_old_logp[j], mb_adv, clip_eps)
                    + vf_coef * value_loss(values, b_ret[j])
                    - ent_coef * dist.entropy().mean()   # a small push to keep exploring
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

    env.close()
    return history
