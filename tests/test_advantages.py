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
        g = (method.group_size or 4) + method.extra_rollouts
        extra = {"seq_logps": -torch.arange(1.0, 2 * g + 1)} if "seq_logps" in method.needs else {}
        if "cluster_ids" in method.needs:
            extra["cluster_ids"] = torch.tensor([0, 1])
        adv, keep = method(_batch([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0][: 2 * g], group_size=g, **extra))
        assert adv.shape == (2 * g,) and adv.dtype == torch.float32 and torch.isfinite(adv).all(), name


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


# ---- Tier 1 methods: the hand-worked examples in docs/method-specs.md, checked to 4 decimals ---------

def _adv(name, rewards, group_size, **params):
    return build_advantage(name, params)(_batch(rewards, group_size))


def _close(got, want):
    torch.testing.assert_close(got, torch.tensor(want, dtype=torch.float32), rtol=0, atol=1e-4)


# The three groups every spec entry works through (group size 4): one right, three right, none right.
ONE, THREE, NONE = [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]


def test_vanilla_spec_example():
    adv, _ = _adv("vanilla", ONE + THREE + NONE, 4)
    _close(adv, [1.4997, -0.4999, -0.4999, -0.4999, 0.4999, 0.4999, 0.4999, -1.4997, 0, 0, 0, 0])


def test_rloo_is_leave_one_out():
    # 2402.14740 sec. 2.3: A_i = r_i - mean of the other G - 1 rewards.
    adv, keep = _adv("rloo", ONE + THREE + NONE, 4)
    _close(adv, [1, -1 / 3, -1 / 3, -1 / 3, 1 / 3, 1 / 3, 1 / 3, -1, 0, 0, 0, 0])
    assert keep is None
    # ... which is G / (G - 1) times Dr. GRPO's advantage, for any rewards (method-specs.md, finding 1).
    r = torch.rand(5 * 6, generator=torch.Generator().manual_seed(1)).tolist()
    rloo, _ = _adv("rloo", r, 6)
    dr, _ = _adv("dr_grpo", r, 6)
    torch.testing.assert_close(rloo, dr * 6 / 5)


def test_sign_ignores_the_group():
    # 2605.07689 eq. 5: A = 2r - 1. All-wrong groups still get -1: that is the point of the method.
    adv, _ = _adv("sign", ONE + THREE + NONE, 4)
    _close(adv, [1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _adv("sign", [2.0, 0.0], 2)


def test_ngrpo_adds_a_virtual_correct_sample():
    # 2509.18851 eqs. 5-6, with the official code's std (n - 1 over G + 1 values) and eps 1e-6.
    adv, _ = _adv("ngrpo", ONE + THREE + NONE, 4)
    _close(adv, [1.0954, -0.7303, -0.7303, -0.7303, 0.4472, 0.4472, 0.4472, -1.7889,
                 -0.4472, -0.4472, -0.4472, -0.4472])
    # The virtual sample takes no gradient, so each group's advantages sum to a negative number.
    assert (adv.view(3, 4).sum(dim=1) < 0).all()
    # An all-correct group: 5 equal values, std 0, so A = 0 (no NaN).
    adv, _ = _adv("ngrpo", [1.0] * 4, 4)
    assert torch.all(adv == 0)


@pytest.mark.parametrize("tau, want", [
    (0.1, [2.9995, -0.9998, -0.9998, -0.9998, 0.3333, 0.3333, 0.3333, -0.9999, 0, 0, 0, 0]),
    (0.3, [2.6133, -0.8711, -0.8711, -0.8711, 0.3177, 0.3177, 0.3177, -0.9530, 0, 0, 0, 0]),
])
def test_softmax_grpo(tau, want):
    # 2608.09271 eq. 1: A = G softmax(r / tau) - 1. Sums to 0 per group, bounded in (-1, G - 1).
    adv, _ = _adv("softmax_grpo", ONE + THREE + NONE, 4, tau=tau)
    _close(adv, want)
    torch.testing.assert_close(adv.view(3, 4).sum(dim=1), torch.zeros(3), atol=1e-5, rtol=0)


def test_softmax_grpo_does_not_overflow_at_a_tiny_tau():
    adv, _ = _adv("softmax_grpo", [1.0, 0.0, 0.0, 0.0], 4, tau=1e-4)
    _close(adv, [3, -1, -1, -1])


def test_corpo_threshold():
    # 2511.04439 eqs. 11-12 with c = 0.5: A = r - max(c, mean).
    adv, _ = _adv("corpo", ONE + THREE + NONE, 4, c=0.5)
    _close(adv, [0.5, -0.5, -0.5, -0.5, 0.25, 0.25, 0.25, -0.75, -0.5, -0.5, -0.5, -0.5])
    # With the paper's own threshold, c = 0, it is exactly Dr. GRPO's advantage on 0/1 rewards.
    r = (torch.rand(40, generator=torch.Generator().manual_seed(2)) > 0.5).float().tolist()
    torch.testing.assert_close(_adv("corpo", r, 8, c=0.0)[0], _adv("dr_grpo", r, 8)[0])


def test_mc_grpo_median_and_mad():
    # 2601.22582 eqs. 2-6 on G + 1 = 5 rewards, eps 1e-3. With 0/1 rewards MAD is 0, so the minority
    # completions get +-1000 and the majority 0; one median completion is dropped from the loss.
    adv, keep = _adv("mc_grpo", [1.0, 0.0, 0.0, 0.0, 1.0] + [1.0, 1.0, 1.0, 0.0, 0.0] + [0.0] * 5, 5)
    _close(adv, [1000, 0, 0, 0, 1000] + [0, 0, 0, -1000, -1000] + [0] * 5)
    assert keep.tolist() == [True, False, True, True, True] + [False, True, True, True, True] + [False] + [True] * 4
    assert (keep.view(3, 5).sum(dim=1) == 4).all(), "G = 4 completions per prompt enter the loss"


def test_mc_grpo_with_a_graded_reward():
    # Their GSM8K reward was 2 / 1.5 / 0, where MAD is not 0. Rewards [2, 1.5, 0, 2, 0]: median 1.5,
    # |r - 1.5| = [0.5, 0, 1.5, 0.5, 1.5], MAD 0.5. The 1.5 completion (the median) is dropped.
    adv, keep = _adv("mc_grpo", [2.0, 1.5, 0.0, 2.0, 0.0], 5)
    _close(adv, [0.5 / 0.501, 0, -1.5 / 0.501, 0.5 / 0.501, -1.5 / 0.501])
    assert keep.tolist() == [True, False, True, True, True]


def test_mc_grpo_asks_for_one_extra_rollout():
    assert build_advantage("mc_grpo").extra_rollouts == 1
    assert build_advantage("vanilla").extra_rollouts == 0


def test_two_grpo_is_vanilla_at_group_size_2_only():
    adv, _ = _adv("two_grpo", [1.0, 0.0, 1.0, 1.0, 0.0, 0.0], 2)
    _close(adv, [0.7070, -0.7070, 0, 0, 0, 0])
    method = build_advantage("two_grpo")
    method.check_group_size(2)
    with pytest.raises(ValueError, match="budget.group_size=2"):
        method.check_group_size(8)
    build_advantage("vanilla").check_group_size(8)     # other methods take any group size


def test_every_method_config_loads_with_its_own_settings():
    from src.config import load_config
    cfg = load_config("ngrpo")
    assert (cfg["loss"]["eps_low"], cfg["loss"]["eps_high"]) == (0.16, 0.24)
    assert load_config("rloo")["loss"]["loss_type"] == "dr_grpo"
    assert load_config("corpo")["advantage"]["params"]["c"] == 0.5
    assert load_config("softmax_grpo", ["advantage.params.tau=0.3"])["advantage"]["params"]["tau"] == 0.3
    for name in ("sign", "softmax_grpo", "corpo", "mc_grpo", "two_grpo"):
        # Methods whose papers change nothing around the advantage inherit the base loss untouched.
        assert load_config(name)["loss"] == load_config("vanilla")["loss"], name


def test_dapo_overlong_penalty():
    # DAPO eq. 13 at L_max 512, L_cache 102: free up to 410 tokens, then linear down to -1 at 512.
    from src.advantages.dapo import overlong_penalty
    got = overlong_penalty(torch.tensor([10, 410, 411, 461, 512]), l_max=512, l_cache=102)
    _close(got, [0, 0, -1 / 102, -51 / 102, -1])
    with pytest.raises(ValueError):
        overlong_penalty(torch.tensor([1]), l_max=100, l_cache=100)


def test_dapo_is_vanilla_on_the_shaped_reward():
    rewards = [1.0, 0.0, 1.0, 0.0]
    lengths = torch.tensor([100, 100, 512, 461])
    batch = AdvantageBatch(rewards=torch.tensor(rewards), group_size=4, prompt_ids=torch.arange(1), lengths=lengths)
    adv, keep = build_advantage("dapo", {"l_max": 512, "l_cache": 102, "overlong_scale": 0.5})(batch)
    shaped = [1.0, 0.0, 1.0 - 0.5, 0.0 - 0.5 * 51 / 102]
    torch.testing.assert_close(adv, group_advantages(shaped, 4))
    assert keep is None
    # Short answers: no shaping, exactly vanilla.
    short = AdvantageBatch(rewards=torch.tensor(rewards), group_size=4, prompt_ids=torch.arange(1),
                           lengths=torch.full((4,), 50))
    torch.testing.assert_close(build_advantage("dapo")(short)[0], group_advantages(rewards, 4))


def test_dapo_config():
    from src.config import load_config
    cfg = load_config("dapo")
    assert cfg["loss"]["loss_type"] == "dapo" and (cfg["loss"]["eps_low"], cfg["loss"]["eps_high"]) == (0.2, 0.28)
    assert cfg["filter"]["drop_uniform_groups"] is True
    assert cfg["advantage"]["params"]["l_max"] == cfg["sampling"]["max_new_tokens"]
    assert load_config("vanilla")["filter"]["drop_uniform_groups"] is False


# ---- Tier 2 methods: worked examples in docs/method-specs-tier2.md -----------------------------------

def _lens(rewards, logps, lengths=(10, 20, 30, 40), **params):
    k = len(rewards) // 4
    batch = AdvantageBatch(rewards=torch.tensor(rewards, dtype=torch.float32), group_size=4,
                           prompt_ids=torch.arange(k), lengths=torch.tensor(list(lengths) * k),
                           seq_logps=torch.tensor(logps, dtype=torch.float32))
    return build_advantage("lens", params)(batch)[0]


P1, P2 = [-10.0, -20.0, -30.0, -40.0], [-5.0, -15.0, -25.0, -35.0]


def test_lens_spec_examples():
    # 2510.08696 sec. 5.1. Equal per-token confidence or one wrong answer: the same as vanilla GRPO.
    _close(_lens([1, 0, 0, 0], P1), [1.4998, -0.4999, -0.4999, -0.4999])
    _close(_lens([1, 1, 1, 0], P2), [0.4999, 0.4999, 0.4999, -1.4997])
    # Wrong answers of different confidence: the most confident (highest per-token prob) is pushed down most.
    adv = _lens([1, 0, 0, 0], P2)
    _close(adv, [1.4994, -0.5280, -0.4933, -0.4781])
    assert adv[1] < adv[2] < adv[3]


def test_lens_trains_on_all_wrong_groups():
    # All wrong, unequal confidence: de-meaned calibrated rewards times alpha = 0.25.
    _close(_lens([0, 0, 0, 0], P2), [-0.0200, 0.0026, 0.0076, 0.0098])
    _close(_lens([0, 0, 0, 0], P2, alpha=1.0), [-0.0800, 0.0105, 0.0304, 0.0391])
    _close(_lens([0, 0, 0, 0], P1), [0, 0, 0, 0])            # equal confidence: nothing to prefer
    _close(_lens([1, 1, 1, 1], P2), [0, 0, 0, 0])            # all right: 0, as vanilla


def test_lens_survives_extreme_log_probs():
    # 1 / pi_old overflows for long answers: D_imp is computed in log space.
    adv = _lens([1, 0, 0, 0], [-5000.0, -20.0, -30.0, -40.0], lengths=(1000, 20, 30, 40))
    assert torch.isfinite(adv).all()


def test_lens_needs_log_probs():
    method = build_advantage("lens")
    assert method.needs == ("seq_logps",)
    with pytest.raises(ValueError, match="seq_logps"):
        method(_batch([1.0, 0.0, 0.0, 0.0], 4))



def _seq(name, steps, **params):
    """Run a stateful method over several steps: [(rewards, prompt_ids, cluster_ids or None), ...]."""
    method = build_advantage(name, params)
    out = []
    for k, (rewards, pids, cids) in enumerate(steps):
        b = AdvantageBatch(rewards=torch.tensor(rewards, dtype=torch.float32), group_size=4,
                           prompt_ids=torch.tensor(pids), lengths=torch.full((len(rewards),), 10), step=k,
                           cluster_ids=None if cids is None else torch.tensor(cids))
        out.append(method(b)[0])
    return out, method.state


def test_ebpo_spec_example():
    # 2602.05165 eqs. 2-4, Algorithm 1: running stats updated with the batch first, then S, then batch norm.
    (a0, a1, a2), st = _seq("ebpo", [([1, 0, 0, 0, 1, 1, 1, 0], [0, 1], None),
                                     ([0, 0, 0, 0, 1, 1, 0, 0], [0, 2], None),
                                     ([0, 0, 0, 0], [0], None)])
    _close(a0, [1.3931, -0.7206, -0.7206, -0.7206, 0.7206, 0.7206, 0.7206, -1.3931])
    # An all-wrong group next to a mixed one is pushed DOWN (vanilla would give it 0).
    _close(a1, [-0.2397, -0.2397, -0.2397, -0.2397, 1.5181, 1.5181, -1.0387, -1.0387])
    _close(a2, [0, 0, 0, 0])                                   # one uniform group: batch std 0, A = 0
    assert (st["glob_n"], st["means_n"]) == (20, 5) and abs(st["S_last"] - 0.3422) < 1e-4


def test_ebpo_global_mean_cancels():
    # After batch centring, mu_glob drops out: A = ((r - mu_g) + S (mu_g - batch mean)) / std.
    method = build_advantage("ebpo")
    r = (torch.rand(32, generator=torch.Generator().manual_seed(4)) > 0.4).float()
    adv, _ = method(AdvantageBatch(rewards=r, group_size=4, prompt_ids=torch.arange(8), lengths=torch.ones(32)))
    S, g = method.state["S_last"], r.view(8, 4).double()
    raw = (g - g.mean(1, keepdim=True) + S * (g.mean(1, keepdim=True) - g.mean())).flatten()
    torch.testing.assert_close(adv, (raw / (raw.std() + 1e-4)).float(), atol=1e-5, rtol=0)


def test_krpo_spec_example():
    # 2505.07527 eqs. 1-6 with the code's x0 = 0, P0 = 1, Q = 1e-5, R = 1e-2, eps = 1e-8. Reset per group.
    (a0, a1), st = _seq("krpo", [([1, 0, 0, 0, 1, 1, 1, 0], [0, 1], None), ([1, 1, 0, 0], [0], None)])
    _close(a0, [0.0995, -7.0481, -5.7514, -4.9725, 0.0995, 0.0705, 0.0575, -14.9376])
    _close(a1, [0.0995, 0.0705, -11.5085, -9.9501])
    assert st == {}, "nothing carries over between steps"
    # Row order matters, and a wrong answer after only wrong answers gets 0.
    (a,), _ = _seq("krpo", [([0, 0, 0, 1, 1, 1, 1, 1], [0, 1], None)])
    _close(a, [0, 0, 0, 14.9873, 0.0995, 0.0705, 0.0575, 0.0497])


def test_ha_dw_spec_example():
    # 2601.08521 eqs. 7-17 at lambda 1.3 and the TRL PR's eta 0.1, m 10; C starts at the first batch's accuracy.
    (a0, a1), st = _seq("ha_dw", [([1, 0, 0, 0, 1, 1, 1, 0], [0, 1], None), ([0, 0, 0, 0, 1, 1, 0, 0], [2, 3], None)],
                        lam=1.3, eta=0.1, m=10)
    _close(a0, [2.5033, -0.5061, -0.5061, -0.5061, 0.5061, 0.5061, 0.5061, -2.5033])
    _close(a1, [0, 0, 0, 0, 1.1256, 1.1256, -1.1256, -1.1256])   # p_hat = C: weight is just lambda
    assert st["C"] == pytest.approx(0.475) and st["hist"] == pytest.approx([0.5, 0.475])
    # Our default eta = 10 gives the same advantages here (they use the prior C) but a faster-moving anchor.
    (b0, b1), st10 = _seq("ha_dw", [([1, 0, 0, 0, 1, 1, 1, 0], [0, 1], None), ([0, 0, 0, 0, 1, 1, 0, 0], [2, 3], None)])
    torch.testing.assert_close(b1, a1)
    assert st10["C"] == 0.25


def test_signbalance_spec_example():
    # 2609.04063 eq. 11: +1 right, -n+/n- wrong, per group; all-wrong 0; all-right +1 as Algorithm 2 is written.
    adv, _ = _adv("signbalance", ONE + THREE + NONE + [1.0] * 4, 4)
    _close(adv, [1, -1 / 3, -1 / 3, -1 / 3, 1, 1, 1, -3, 0, 0, 0, 0, 1, 1, 1, 1])
    torch.testing.assert_close(adv.view(4, 4)[:3].sum(dim=1), torch.zeros(3))
    zero, _ = _adv("signbalance", [1.0] * 4, 4, all_correct="zero")
    _close(zero, [0, 0, 0, 0])


def test_bv_blend_spec_example():
    # 2606.28707 eqs. 5-12. Step 0: cluster 0 unseen, w = 0, exactly vanilla; then initialised from the
    # pooled rollouts. Step 1: an all-wrong group in that cluster gets a NEGATIVE advantage (vanilla: 0).
    (a0, a1), st = _seq("bv_blend", [([1, 0, 0, 0, 1, 1, 1, 0], [0, 1], [0, 0]), ([0, 0, 0, 0], [2], [0])])
    _close(a0, [1.4997, -0.4999, -0.4999, -0.4999, 0.4999, 0.4999, 0.4999, -1.4997])
    _close(a1, [-0.4344, -0.4344, -0.4344, -0.4344])
    assert abs(st["m1"][0].item() - 0.05) < 1e-12 and abs(st["n_eff"][0].item() - 4.4) < 1e-12
    assert st["seen"].tolist()[:2] == [True, False]


def test_bv_blend_needs_clusters():
    with pytest.raises(ValueError, match="cluster_ids"):
        build_advantage("bv_blend")(_batch([1.0, 0.0, 0.0, 0.0], 4))
    with pytest.raises(ValueError, match="cluster ids must be"):
        build_advantage("bv_blend", {"n_clusters": 2})(_batch([1.0, 0.0, 0.0, 0.0], 4, cluster_ids=torch.tensor([5])))
