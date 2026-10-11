"""Checks for src/advantages/grpo.py, vanilla GRPO's advantage.

From week 11 each new method gets its own tests in this file, against a hand-worked example from its paper.
"""
import math

import torch

from src.advantages.grpo import group_advantages


def test_hand_computed_example():
    adv = group_advantages([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0], group_size=4)
    # group 1: mean 0.5, sample std sqrt(1/3); group 2: mean 0.75, sample std 0.5
    s1, s2 = math.sqrt(1 / 3) + 1e-4, 0.5 + 1e-4
    want = [0.5 / s1, -0.5 / s1, -0.5 / s1, 0.5 / s1, 0.25 / s2, 0.25 / s2, 0.25 / s2, -0.75 / s2]
    torch.testing.assert_close(
        adv, torch.tensor(want), rtol=1e-5, atol=1e-6,
        msg="A = (r - group mean) / (group std + 1e-4), with the sample std (Bessel's correction), like TRL",
    )


def test_groups_are_consecutive_rows():
    adv = group_advantages([1.0, 1.0, 0.0, 0.0], group_size=2)
    assert torch.all(adv == 0), "rows 0-1 are one group and rows 2-3 another; both are uniform, so A = 0"


def test_a_uniform_group_gives_zeros_not_nan():
    adv = group_advantages([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], group_size=4)
    assert torch.all(adv == 0), "no learning signal from an all-correct or all-wrong group, and no NaN"


def test_scale_none_only_subtracts_the_mean():
    adv = group_advantages([1.0, 0.0, 0.0, 0.0], group_size=4, scale="none")
    torch.testing.assert_close(adv, torch.tensor([0.75, -0.25, -0.25, -0.25]), msg="Dr. GRPO: no std division")


def test_advantages_sum_to_zero_in_every_group():
    rewards = torch.rand(6 * 8, generator=torch.Generator().manual_seed(0))
    adv = group_advantages(rewards, group_size=8)
    assert adv.shape == (48,) and adv.dtype == torch.float32
    torch.testing.assert_close(adv.view(6, 8).sum(dim=1), torch.zeros(6), atol=1e-5, rtol=0)
