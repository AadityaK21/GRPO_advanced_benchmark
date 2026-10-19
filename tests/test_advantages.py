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


# ---- The method interface (src/advantages/base.py) and the methods ---------------------------------

import pytest

from src.advantages.base import AdvantageBatch, build_advantage


def _batch(rewards, group_size, **kw):
    r = torch.tensor(rewards, dtype=torch.float32)
    n = r.numel() // group_size
    return AdvantageBatch(rewards=r, group_size=group_size, prompt_ids=torch.arange(n),
                          lengths=torch.full((r.numel(),), 10), **kw)


def test_every_method_file_follows_the_interface():
    """Each method in configs/methods/ loads, and returns one finite advantage per completion."""
    from pathlib import Path
    names = sorted(p.stem for p in (Path(__file__).resolve().parents[1] / "configs" / "methods").glob("*.yaml"))
    assert "vanilla" in names and "dr_grpo" in names
    import yaml
    for name in names:
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "methods" / f"{name}.yaml").read_text())
        method = build_advantage(cfg["advantage"]["name"], cfg["advantage"].get("params"))
        adv, keep = method(_batch([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], group_size=4))
        assert adv.shape == (8,) and adv.dtype == torch.float32 and torch.isfinite(adv).all(), name


def test_vanilla_is_the_reference_group_advantage():
    rewards = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    adv, keep = build_advantage("vanilla")(_batch(rewards, 4))
    torch.testing.assert_close(adv, group_advantages(rewards, 4))
    assert keep is None


def test_dr_grpo_hand_worked_example():
    # Dr. GRPO (2503.20783, sec. 3.2): A = r - mean(r), no division by the group's std.
    # Group 1: 1 of 4 right, mean 0.25. Group 2: 3 of 4 right, mean 0.75.
    adv, _ = build_advantage("dr_grpo")(_batch([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], 4))
    torch.testing.assert_close(adv, torch.tensor([0.75, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, -0.75]))
    # Vanilla divides by the group's sample std, 0.5 for group 1, so its one right answer gets
    # 0.75 / 0.5 = 1.5: a hard prompt's rare success is scaled up. Dr. GRPO leaves it at 0.75.
    vanilla, _ = build_advantage("vanilla")(_batch([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], 4))
    assert vanilla[0] > 1.4 and adv[0] == 0.75


def test_state_persists_across_calls_and_params_are_bound():
    from src.advantages.base import Advantage

    def counting(batch, state, offset=0.0):
        state["calls"] = state.get("calls", 0) + 1
        return batch.rewards * 0 + offset + state["calls"]

    method = Advantage(name="counting", fn=counting, params={"offset": 10.0})
    first, _ = method(_batch([1.0, 0.0], 2))
    second, _ = method(_batch([1.0, 0.0], 2))
    assert first.tolist() == [11.0, 11.0] and second.tolist() == [12.0, 12.0]


def test_bad_methods_are_caught():
    from src.advantages.base import Advantage
    with pytest.raises(ValueError, match="no advantage method"):
        build_advantage("no_such_method")
    with pytest.raises(ValueError):
        build_advantage("grpo")          # the helper module, not a method
    with pytest.raises(ValueError, match="shape"):
        Advantage(name="short", fn=lambda b, s: b.rewards[:1])(_batch([1.0, 0.0], 2))
    with pytest.raises(FloatingPointError):
        Advantage(name="nan", fn=lambda b, s: b.rewards / 0 * 0)(_batch([1.0, 0.0], 2))
    with pytest.raises(ValueError, match="divisible"):
        _batch([1.0, 0.0, 1.0], 2)
