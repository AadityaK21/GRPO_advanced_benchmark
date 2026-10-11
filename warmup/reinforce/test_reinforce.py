"""Checks for reinforce.py.

These tests check properties, not one particular implementation, so any
correct REINFORCE passes. If a test looks wrong to you, write down why
before you change anything.
"""
import math
import time

import numpy as np
import pytest
import torch
import torch.nn as nn

import reinforce as R


# ---------------------------------------------------------------- returns


def test_returns_hand_computed():
    out = R.discounted_returns([1.0, 1.0, 1.0], 0.9)
    np.testing.assert_allclose(out, [2.71, 1.9, 1.0], atol=1e-9)


def test_returns_gamma_one_and_zero():
    np.testing.assert_allclose(R.discounted_returns([1.0, 2.0, 3.0], 1.0), [6.0, 5.0, 3.0])
    np.testing.assert_allclose(R.discounted_returns([1.0, 2.0, 3.0], 0.0), [1.0, 2.0, 3.0])


def test_returns_shape_dtype_and_input_untouched():
    rewards = [0.5, -1.0, 2.0, 0.0]
    before = list(rewards)
    out = R.discounted_returns(rewards, 0.99)
    assert isinstance(out, np.ndarray) and out.shape == (4,)
    assert np.issubdtype(out.dtype, np.floating)
    assert rewards == before, "discounted_returns must not modify its input"


def test_returns_match_brute_force():
    rewards = np.random.default_rng(0).normal(size=50)
    gamma = 0.97
    brute = [sum(gamma ** (k - t) * rewards[k] for k in range(t, 50)) for t in range(50)]
    np.testing.assert_allclose(R.discounted_returns(rewards, gamma), brute, rtol=1e-10)


def test_returns_take_linear_time():
    rewards = np.ones(5000)
    start = time.perf_counter()
    R.discounted_returns(rewards, 0.99)
    assert time.perf_counter() - start < 0.5, "use one backward pass: G[t] = r[t] + gamma * G[t+1]"


# ---------------------------------------------------- policy and sampling


class FixedLogits(nn.Module):
    """A stand-in policy that ignores the observation and returns fixed logits."""

    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, obs):
        return self.logits.expand(obs.shape[0], -1)


def test_policy_output_shape():
    torch.manual_seed(0)
    policy = R.Policy(4, 2)
    assert policy(torch.zeros(8, 4)).shape == (8, 2)


def test_select_action_returns_int_and_attached_log_prob():
    action, log_prob = R.select_action(FixedLogits([0.0, 0.0]), np.zeros(4, dtype=np.float32))
    assert isinstance(action, int) and action in (0, 1)
    assert log_prob.dim() == 0, "log_prob must be a 0-dim tensor for one action"
    assert log_prob.requires_grad, "log_prob must stay attached to the autograd graph"
    assert math.isclose(log_prob.item(), math.log(0.5), abs_tol=1e-6)


def test_select_action_samples_in_proportion():
    torch.manual_seed(0)
    probs = torch.tensor([0.7, 0.2, 0.1])
    policy = FixedLogits(probs.log().tolist())
    counts = np.zeros(3)
    for _ in range(4000):
        action, _ = R.select_action(policy, np.zeros(4, dtype=np.float32))
        counts[action] += 1
    np.testing.assert_allclose(
        counts / counts.sum(), probs.numpy(), atol=0.03,
        err_msg="sample from the policy's distribution; don't take the argmax",
    )


def test_select_action_log_prob_belongs_to_the_sampled_action():
    torch.manual_seed(1)
    probs = torch.tensor([0.6, 0.3, 0.1])
    policy = FixedLogits(probs.log().tolist())
    for _ in range(50):
        action, log_prob = R.select_action(policy, np.zeros(4, dtype=np.float32))
        assert math.isclose(log_prob.item(), math.log(probs[action].item()), abs_tol=1e-5)


# ----------------------------------------------------------- the loss


def test_loss_value_without_baseline():
    log_probs = torch.tensor([math.log(0.5), math.log(0.25)])
    returns = torch.tensor([2.0, 1.0])
    loss = R.reinforce_loss(log_probs, returns)
    assert math.isclose(loss.item(), 2 * math.log(2), rel_tol=1e-6), "loss = -mean(log_prob * advantage)"


def test_loss_value_with_baseline():
    log_probs = torch.tensor([math.log(0.5), math.log(0.25)])
    returns = torch.tensor([2.0, 1.0])
    expected = -(math.log(0.5) * 0.5 + math.log(0.25) * -0.5) / 2
    assert math.isclose(R.reinforce_loss(log_probs, returns, 1.5).item(), expected, rel_tol=1e-6)
    tensor_baseline = torch.tensor([1.5, 1.5])
    assert math.isclose(R.reinforce_loss(log_probs, returns, tensor_baseline).item(), expected, rel_tol=1e-6)


def test_no_gradient_flows_into_returns_or_baseline():
    log_probs = torch.tensor([-0.5, -1.0], requires_grad=True)
    returns = torch.tensor([1.0, 2.0], requires_grad=True)
    baseline = torch.tensor([0.5, 0.5], requires_grad=True)
    R.reinforce_loss(log_probs, returns, baseline).backward()
    assert log_probs.grad is not None
    assert returns.grad is None and baseline.grad is None, (
        "detach the advantage: returns and baseline are constants, not things to optimise"
    )


@pytest.mark.parametrize("advantage, direction", [(1.0, 1), (-1.0, -1)])
def test_gradient_step_moves_log_prob_the_right_way(advantage, direction):
    theta = torch.zeros(3, requires_grad=True)
    log_probs = torch.log_softmax(theta, dim=0)[[0]]
    R.reinforce_loss(log_probs, torch.tensor([advantage])).backward()
    with torch.no_grad():
        after = torch.log_softmax(theta - 0.1 * theta.grad, dim=0)[0]
        before = torch.log_softmax(torch.zeros(3), dim=0)[0]
    assert (after - before).item() * direction > 0, (
        "a positive advantage must make its action more likely, a negative one less likely"
    )


# --------------------------------------------- the baseline, on a bandit

MEANS = torch.tensor([1.0, 2.0, 3.0])
THETA0 = torch.tensor([0.5, 0.0, -0.5])


def _batch_gradients(baseline, n_batches=400, batch=64, seed=0):
    """One REINFORCE estimate of d E[reward] / d logits per batch, on a 3-armed bandit."""
    gen = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(n_batches):
        theta = THETA0.clone().requires_grad_(True)
        probs = torch.softmax(theta, dim=0).detach()
        actions = torch.multinomial(probs, batch, replacement=True, generator=gen)
        rewards = MEANS[actions] + torch.randn(batch, generator=gen)
        log_probs = torch.log_softmax(theta, dim=0)[actions]
        R.reinforce_loss(log_probs, rewards, baseline).backward()
        estimates.append(-theta.grad.clone())  # the loss gradient is minus the estimate
    return torch.stack(estimates)


def _true_gradient():
    p = torch.softmax(THETA0, dim=0)
    return p * (MEANS - (p * MEANS).sum())


@pytest.mark.parametrize("baseline", [None, 2.0])
def test_estimate_is_unbiased_with_or_without_baseline(baseline):
    estimates = _batch_gradients(baseline)
    np.testing.assert_allclose(
        estimates.mean(0).numpy(), _true_gradient().numpy(), atol=0.03,
        err_msg="a constant baseline must not change the expected gradient",
    )


def test_baseline_reduces_variance():
    expected_reward = (torch.softmax(THETA0, dim=0) * MEANS).sum().item()
    spread_without = _batch_gradients(None).var(0).sum()
    spread_with = _batch_gradients(expected_reward).var(0).sum()
    assert spread_with < 0.6 * spread_without, "subtracting E[reward] should cut the variance a lot"


# ------------------------------------------------------------ training


def test_train_is_reproducible():
    first = R.train(episodes=20, seed=123)
    second = R.train(episodes=20, seed=123)
    assert len(first) == 20
    assert first == second, "seed torch, numpy and the environment so one seed gives one run"


@pytest.mark.slow
def test_cartpole_learns():
    learned = 0
    for seed in (0, 1, 2):
        returns = R.train(episodes=600, seed=seed)
        assert len(returns) == 600
        first, last = np.mean(returns[:50]), np.mean(returns[-50:])
        print(f"seed {seed}: first 50 episodes {first:.0f}, last 50 episodes {last:.0f}")
        if last > 150 and last > 3 * first:
            learned += 1
    assert learned >= 2, "REINFORCE should clearly learn CartPole on at least 2 of 3 seeds"
