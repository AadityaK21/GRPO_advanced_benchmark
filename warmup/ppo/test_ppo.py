"""Checks for ppo.py.

These tests check properties, not one particular implementation, so any
correct PPO passes. If a test looks wrong to you, write down why before
you change anything.
"""
import math

import numpy as np
import pytest
import torch

import ppo as P


# -------------------------------------------------------------------- GAE


def _gae_brute_force(rewards, values, dones, last_value, gamma, lam):
    """A[t] = sum over k >= t of (gamma*lam)^(k-t) * delta[k], stopping after an episode ends."""
    T = len(rewards)
    next_values = np.append(values[1:], last_value)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    adv = np.zeros(T)
    for t in range(T):
        total, discount = 0.0, 1.0
        for k in range(t, T):
            total += discount * deltas[k]
            if dones[k]:
                break
            discount *= gamma * lam
        adv[t] = total
    return adv, adv + values


def test_gae_hand_computed_no_dones():
    adv, ret = P.compute_gae(np.ones(3), np.zeros(3), np.zeros(3), 0.0, 1.0, 1.0)
    np.testing.assert_allclose(adv, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(ret, [3.0, 2.0, 1.0])


def test_gae_stops_at_episode_end():
    adv, _ = P.compute_gae(np.ones(3), np.zeros(3), np.array([0.0, 1.0, 0.0]), 0.0, 1.0, 1.0)
    np.testing.assert_allclose(adv, [2.0, 1.0, 1.0], err_msg="nothing may flow back across an episode end")


def test_gae_uses_last_value_to_bootstrap():
    values = np.array([0.0, 0.0, 0.0])
    adv, _ = P.compute_gae(np.ones(3), values, np.zeros(3), 10.0, 0.5, 0.0)
    np.testing.assert_allclose(adv, [1.0, 1.0, 6.0], err_msg="the last step bootstraps from last_value")


def test_gae_lambda_zero_is_one_step_td_error():
    rng = np.random.default_rng(1)
    rewards, values = rng.normal(size=6), rng.normal(size=6)
    dones = np.array([0, 0, 1, 0, 0, 0], dtype=float)
    adv, _ = P.compute_gae(rewards, values, dones, 0.7, 0.9, 0.0)
    next_values = np.append(values[1:], 0.7)
    np.testing.assert_allclose(adv, rewards + 0.9 * next_values * (1 - dones) - values, rtol=1e-10)


def test_gae_matches_brute_force_and_returns_add_values():
    rng = np.random.default_rng(0)
    T = 40
    rewards, values = rng.normal(size=T), rng.normal(size=T)
    dones = (rng.random(T) < 0.15).astype(float)
    adv, ret = P.compute_gae(rewards, values, dones, 0.3, 0.99, 0.95)
    want_adv, want_ret = _gae_brute_force(rewards, values, dones, 0.3, 0.99, 0.95)
    np.testing.assert_allclose(adv, want_adv, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ret, want_ret, rtol=1e-8, atol=1e-10, err_msg="returns = advantages + values")


# ----------------------------------------------------------- the network


def test_actor_critic_shapes():
    torch.manual_seed(0)
    logits, values = P.ActorCritic(4, 2)(torch.zeros(8, 4))
    assert logits.shape == (8, 2)
    assert values.shape == (8,), "return values as shape [B], not [B, 1]"


# ------------------------------------------------------- the policy loss


def _loss_grad(ratio, advantage, eps=0.2):
    new = torch.tensor([math.log(ratio)], requires_grad=True)
    loss = P.ppo_policy_loss(new, torch.zeros(1), torch.tensor([advantage]), eps)
    loss.backward()
    return loss.item(), new.grad.item()


def test_policy_loss_at_ratio_one_is_minus_mean_advantage():
    new = torch.zeros(2)
    loss = P.ppo_policy_loss(new, torch.zeros(2), torch.tensor([1.0, -2.0]), 0.2)
    assert math.isclose(loss.item(), 0.5, rel_tol=1e-6), "with ratio 1, loss = -mean(A)"


def test_policy_loss_hand_computed_with_clipping():
    new = torch.tensor([math.log(1.5), math.log(0.5)])
    adv = torch.tensor([1.0, 1.0])
    loss = P.ppo_policy_loss(new, torch.zeros(2), adv, 0.2)
    assert math.isclose(loss.item(), -(1.2 + 0.5) / 2, rel_tol=1e-6)


@pytest.mark.parametrize("ratio, advantage", [(1.5, 1.0), (0.5, -1.0)])
def test_clipping_stops_the_gradient_once_the_update_went_far_enough(ratio, advantage):
    _, grad = _loss_grad(ratio, advantage)
    assert grad == 0.0, "past the clip range in the helpful direction, the gradient must be zero"


@pytest.mark.parametrize("ratio, advantage", [(1.5, -1.0), (0.5, 1.0)])
def test_clipping_never_hides_a_move_in_the_wrong_direction(ratio, advantage):
    _, grad = _loss_grad(ratio, advantage)
    assert grad != 0.0, "min() keeps the unclipped term when the policy moved the wrong way"


@pytest.mark.parametrize("advantage, direction", [(1.0, 1), (-1.0, -1)])
def test_policy_gradient_has_the_right_sign(advantage, direction):
    _, grad = _loss_grad(1.0, advantage)
    assert -grad * direction > 0, "a positive advantage must push its log-prob up"


def test_no_gradient_into_old_log_probs_or_advantages():
    new = torch.tensor([-0.5, -1.0], requires_grad=True)
    old = torch.tensor([-0.6, -0.9], requires_grad=True)
    adv = torch.tensor([1.0, -1.0], requires_grad=True)
    P.ppo_policy_loss(new, old, adv, 0.2).backward()
    assert new.grad is not None
    assert old.grad is None and adv.grad is None, "old log-probs and advantages are constants"


# -------------------------------------------------------- the value loss


def test_value_loss_value_and_no_gradient_into_returns():
    values = torch.tensor([1.0, 2.0], requires_grad=True)
    returns = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = P.value_loss(values, returns)
    assert math.isclose(loss.item(), 1.25, rel_tol=1e-6), "0.5 * mean((values - returns)^2)"
    loss.backward()
    assert values.grad is not None and returns.grad is None


# ------------------------------------------------------------- training


def test_train_is_reproducible():
    first = P.train(total_steps=1024, seed=7)
    second = P.train(total_steps=1024, seed=7)
    assert len(first) > 0
    assert first == second, "seed torch, numpy and the environment so one seed gives one run"


@pytest.mark.slow
def test_cartpole_learns():
    learned = 0
    for seed in (0, 1, 2):
        returns = P.train(total_steps=100_000, seed=seed)
        first, last = np.mean(returns[:20]), np.mean(returns[-20:])
        print(f"seed {seed}: first 20 episodes {first:.0f}, last 20 episodes {last:.0f}")
        if last > 150 and last > 3 * first:
            learned += 1
    assert learned >= 2, "PPO should clearly learn CartPole on at least 2 of 3 seeds"
