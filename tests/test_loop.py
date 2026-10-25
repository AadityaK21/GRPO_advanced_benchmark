"""Checks for src/train/loop.py: one GRPO step, start to finish."""
import copy

import pytest
import torch

import src.train.loop
import src.advantages.base
import src.advantages.grpo
import src.train.sample
from src.advantages.grpo import group_advantages
from src.train.logprobs import per_token_logps
from src.train.loop import GRPOSettings, grpo_step
from src.train.loss import grpo_loss
from tiny import EOS, PAD, make_model, make_tokenizer, tid


def _number(word):
    return int(word[1:]) if word[:1] == "t" and word[1:].isdigit() else None


def _first_token_below_16(texts, answers):
    """Reward 1 if the completion starts with one of t0..t15. A quarter of the vocabulary, so easy to hit."""
    rewards = []
    for text in texts:
        words = text.split()
        k = _number(words[0]) if words else None
        rewards.append(1.0 if k is not None and k < 16 else 0.0)
    return rewards


def _prob_first_token_below_16(model, tokenizer, prompts):
    with torch.no_grad():
        enc = tokenizer(prompts, return_tensors="pt")
        p = torch.softmax(model(**enc).logits[:, -1], dim=-1)
    return p[:, tid(0):tid(16)].sum(dim=-1).mean().item()


def test_rewards_see_each_answer_next_to_its_own_completions():
    model, tokenizer = make_model(), make_tokenizer()
    seen = {}

    def reward_fn(texts, answers):
        seen["texts"], seen["answers"] = list(texts), list(answers)
        return [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]

    torch.manual_seed(0)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    stats = grpo_step(model, opt, tokenizer, ["t10", "t11"], ["A", "B"], reward_fn,
                      GRPOSettings(group_size=3, max_new_tokens=4))
    assert seen["answers"] == ["A", "A", "A", "B", "B", "B"], "completions i*G..i*G+G-1 belong to prompt i, and so does its answer"
    assert len(seen["texts"]) == 6 and all(isinstance(t, str) for t in seen["texts"])
    assert abs(stats["reward"] - 0.5) < 1e-6, "stats['reward'] is the mean reward of the sampled completions"


def test_equal_rewards_leave_the_model_unchanged():
    model, tokenizer = make_model(), make_tokenizer()
    before = copy.deepcopy(model.state_dict())
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    torch.manual_seed(0)
    grpo_step(model, opt, tokenizer, ["t10", "t11"], ["", ""], lambda texts, answers: [1.0] * len(texts),
              GRPOSettings(group_size=4, max_new_tokens=4))
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), "all advantages are 0, so the gradient must be exactly 0"


def test_one_step_makes_rewarded_tokens_more_likely():
    model, tokenizer = make_model(), make_tokenizer()
    prompts = ["t30", "t31"]
    before = _prob_first_token_below_16(model, tokenizer, prompts)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(0)
    grpo_step(model, opt, tokenizer, prompts, ["", ""], _first_token_below_16, GRPOSettings(group_size=32, max_new_tokens=1))
    after = _prob_first_token_below_16(model, tokenizer, prompts)
    assert after > before, f"P(rewarded first token) went {before:.3f} -> {after:.3f}; check every sign"


def test_first_update_is_on_policy_and_later_updates_clip():
    tokenizer = make_tokenizer()
    clip, weights = {}, {}
    for updates, eps in ((1, 0.2), (4, 0.2), (4, 100.0)):
        model = make_model()
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        torch.manual_seed(0)
        settings = GRPOSettings(group_size=16, max_new_tokens=4, num_updates=updates, eps_low=eps, eps_high=eps)
        stats = grpo_step(model, opt, tokenizer, ["t30", "t31"], ["", ""], _first_token_below_16, settings)
        clip[updates, eps], weights[updates, eps] = stats["clip_frac"], model.lm_head.weight.detach().clone()
    assert clip[1, 0.2] == 0.0, "with one update the ratio is exactly 1, so nothing can be clipped"
    assert clip[4, 0.2] > 0.0, "old log-probs are computed once, before the first update, and reused by every update"
    assert clip[4, 100.0] == 0.0, "use the settings' clip range"
    assert not torch.equal(weights[4, 0.2], weights[4, 100.0]), "eps_low and eps_high must reach the loss"


def test_kl_to_the_reference_starts_at_zero_then_grows():
    model, tokenizer = make_model(), make_tokenizer()
    ref = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    settings = GRPOSettings(group_size=16, max_new_tokens=4, beta=0.04)
    torch.manual_seed(0)
    kl = [grpo_step(model, opt, tokenizer, ["t30", "t31"], ["", ""], _first_token_below_16, settings, ref_model=ref)["kl"]
          for _ in range(3)]
    assert abs(kl[0]) < 1e-6, "before any update the model is the reference"
    assert kl[2] > 0.0, "the reference stays frozen while the model moves away from it"


FIXED_BATCH = {
    "prompt_ids": torch.tensor([[PAD, 12, 13]] * 4 + [[14, 15, 16]] * 4),
    "prompt_mask": torch.tensor([[0, 1, 1]] * 4 + [[1, 1, 1]] * 4),
    "completion_ids": torch.tensor([[20, 21, EOS, PAD], [22, 23, 24, 25], [26, EOS, PAD, PAD], [27, 28, 29, 30],
                                    [31, 32, 33, EOS], [34, 35, 36, 37], [EOS, PAD, PAD, PAD], [38, 39, 40, 41]]),
    "completion_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1],
                                     [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]]),
}
FIXED_REWARDS = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]


@pytest.mark.parametrize("settings", [
    GRPOSettings(group_size=4, max_new_tokens=8, temperature=0.7, loss_type="grpo", scale="group"),
    GRPOSettings(group_size=4, max_new_tokens=8, temperature=1.3, loss_type="dr_grpo", scale="none", beta=0.04),
])
def test_step_passes_every_setting_to_the_right_piece(monkeypatch, settings):
    """With sampling replaced by a fixed batch, one SGD update must equal -lr times the gradient of your own loss."""
    def fixed_sample_group(model, tokenizer, prompts, group_size, max_new_tokens, temperature):
        return {k: v.clone() for k, v in FIXED_BATCH.items()}

    monkeypatch.setattr(src.train.sample, "sample_group", fixed_sample_group)
    monkeypatch.setattr(src.train.loop, "sample_group", fixed_sample_group, raising=False)
    tokenizer, model = make_tokenizer(), make_model(logit_scale=5.0)
    ref_model = make_model(seed=1, logit_scale=5.0)

    # What one update should do, from your own pieces.
    expected = copy.deepcopy(model)
    b = FIXED_BATCH
    input_ids = torch.cat([b["prompt_ids"], b["completion_ids"]], dim=1)
    attention_mask = torch.cat([b["prompt_mask"], b["completion_mask"]], dim=1)
    k, t = b["completion_ids"].size(1), settings.temperature
    with torch.no_grad():
        ref = per_token_logps(ref_model, input_ids, attention_mask, k, t) if settings.beta else None
    loss = grpo_loss(per_token_logps(expected, input_ids, attention_mask, k, t), None,
                     group_advantages(FIXED_REWARDS, settings.group_size, settings.scale), b["completion_mask"],
                     ref_logps=ref, beta=settings.beta, eps_low=settings.eps_low, eps_high=settings.eps_high,
                     loss_type=settings.loss_type, max_completion_length=settings.max_new_tokens)
    loss.backward()

    before = copy.deepcopy(model.state_dict())
    grpo_step(model, torch.optim.SGD(model.parameters(), lr=1.0), tokenizer, ["p0", "p1"], ["a0", "a1"],
              lambda texts, answers: FIXED_REWARDS, settings, ref_model=ref_model)
    for name, p in expected.named_parameters():
        torch.testing.assert_close(
            before[name] - model.state_dict()[name], p.grad, rtol=1e-4, atol=1e-6,
            msg=f"{name}: check that temperature, scale, loss_type, beta, the clip range, max_new_tokens "
                "and ref_model each reach the piece that uses them",
        )


def test_step_is_reproducible_under_a_seed():
    tokenizer = make_tokenizer()
    results = []
    for _ in range(2):
        model = make_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(5)
        stats = grpo_step(model, opt, tokenizer, ["t30", "t31"], ["", ""], _first_token_below_16,
                          GRPOSettings(group_size=8, max_new_tokens=4, num_updates=2))
        results.append((stats["reward"], stats["clip_frac"], model.lm_head.weight.detach().clone()))
    assert results[0][:2] == results[1][:2] and torch.equal(results[0][2], results[1][2])


def test_learns_a_toy_task():
    """End to end: prompt t40+k must be answered with a token from t(8k)..t(8k+7). Chance is 1/8."""
    tokenizer = make_tokenizer()
    prompts, answers = [f"t{40 + k}" for k in range(4)], ["0", "1", "2", "3"]

    def reward_fn(texts, answers):
        out = []
        for text, answer in zip(texts, answers):
            words = text.split()
            k = _number(words[0]) if words else None
            out.append(1.0 if k is not None and k // 8 == int(answer) else 0.0)
        return out

    learned = 0
    for seed in range(3):
        torch.manual_seed(seed)
        model = make_model(seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        rewards = [grpo_step(model, opt, tokenizer, prompts, answers, reward_fn,
                             GRPOSettings(group_size=8, max_new_tokens=2))["reward"] for _ in range(100)]
        first, last = sum(rewards[:10]) / 10, sum(rewards[-10:]) / 10
        print(f"seed {seed}: mean reward {first:.2f} over the first 10 steps, {last:.2f} over the last 10")
        learned += last > 0.8
    assert learned >= 2, "GRPO should learn this on at least 2 of 3 seeds within 100 steps"


# ---- Harness additions: micro-batches, the advantage interface, diagnostics -------------------------

from src.advantages.base import Advantage, build_advantage  # noqa: E402


def _fixed(monkeypatch, seen=None):
    def fixed_sample_group(model, tokenizer, prompts, group_size, max_new_tokens, temperature):
        if seen is not None:
            seen["group_size"] = group_size
        return {k: v.clone() for k, v in FIXED_BATCH.items()}
    monkeypatch.setattr(src.train.loop, "sample_group", fixed_sample_group)


def _update(monkeypatch, settings, rewards=FIXED_REWARDS, **kw):
    """The parameter change from one SGD step (lr 1) on the fixed batch, and the step's stats."""
    _fixed(monkeypatch)
    model = make_model(logit_scale=5.0)
    before = copy.deepcopy(model.state_dict())
    stats = grpo_step(model, torch.optim.SGD(model.parameters(), lr=1.0), make_tokenizer(), ["p0", "p1"],
                      ["a0", "a1"], lambda texts, answers: rewards, settings,
                      ref_model=make_model(seed=1, logit_scale=5.0) if settings.beta else None, **kw)
    return {n: before[n] - v for n, v in model.state_dict().items()}, stats


@pytest.mark.parametrize("loss_type", ["grpo", "dapo", "bnpo", "dr_grpo"])
@pytest.mark.parametrize("micro", [1, 3, 5])
def test_micro_batches_give_the_full_batch_update(monkeypatch, loss_type, micro):
    kw = dict(group_size=4, max_new_tokens=8, temperature=0.9, loss_type=loss_type, beta=0.04, num_updates=2,
              max_grad_norm=None)
    full, s_full = _update(monkeypatch, GRPOSettings(**kw))
    split, s_split = _update(monkeypatch, GRPOSettings(**kw, micro_batch_size=micro))
    for name in full:
        torch.testing.assert_close(split[name], full[name], rtol=1e-4, atol=1e-6,
                                   msg=f"{name}: weight each micro-batch so the summed gradient is the batch's")
    for key in ("loss", "clip_frac", "kl", "entropy", "grad_norm"):
        assert s_split[key] == pytest.approx(s_full[key], rel=1e-4, abs=1e-6), key


def test_the_vanilla_method_object_is_the_built_in_path(monkeypatch):
    s = GRPOSettings(group_size=4, max_new_tokens=8, loss_type="grpo")
    plain, _ = _update(monkeypatch, s)
    via_method, _ = _update(monkeypatch, s, advantage=build_advantage("vanilla"))
    for name in plain:
        torch.testing.assert_close(via_method[name], plain[name])
    dr, _ = _update(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8, loss_type="dr_grpo"),
                    advantage=build_advantage("dr_grpo"))
    dr_plain, _ = _update(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8, loss_type="dr_grpo", scale="none"))
    for name in dr:
        torch.testing.assert_close(dr[name], dr_plain[name])


def test_methods_get_what_they_ask_for(monkeypatch):
    seen = {}

    def spy(batch, state):
        seen["batch"] = batch
        return torch.zeros_like(batch.rewards)

    method = Advantage(name="spy", fn=spy, needs=("seq_logps",))
    _update(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8, temperature=0.9), advantage=method,
            prompt_ids=[1234, 99], step=7)
    b = seen["batch"]
    assert b.prompt_ids.tolist() == [1234, 99] and b.step == 7 and b.group_size == 4
    assert b.lengths.tolist() == FIXED_BATCH["completion_mask"].sum(dim=1).tolist()
    model = make_model(logit_scale=5.0)
    ids = torch.cat([FIXED_BATCH["prompt_ids"], FIXED_BATCH["completion_ids"]], dim=1)
    am = torch.cat([FIXED_BATCH["prompt_mask"], FIXED_BATCH["completion_mask"]], dim=1)
    with torch.no_grad():
        want = (per_token_logps(model, ids, am, 4, 0.9) * FIXED_BATCH["completion_mask"]).sum(dim=1)
    torch.testing.assert_close(b.seq_logps, want, msg="seq_logps: each completion's summed log-prob, counted tokens only")


def test_extra_rollouts_are_sampled_and_counted(monkeypatch):
    seen = {}
    _fixed(monkeypatch, seen)
    method = Advantage(name="plus_one", fn=lambda b, s: torch.zeros_like(b.rewards), extra_rollouts=1)
    model = make_model()
    # FIXED_BATCH has 8 rows = 2 prompts x 4, so ask for group size 3 plus 1 extra.
    stats = grpo_step(model, torch.optim.SGD(model.parameters(), lr=0.0), make_tokenizer(), ["p0", "p1"],
                      ["a0", "a1"], lambda t, a: FIXED_REWARDS, GRPOSettings(group_size=3, max_new_tokens=8),
                      advantage=method)
    assert seen["group_size"] == 4 and stats["rollouts"] == 8


def test_dropped_completions_give_no_gradient(monkeypatch):
    s = GRPOSettings(group_size=4, max_new_tokens=8, loss_type="dapo")
    adv = group_advantages(FIXED_REWARDS, 4)
    keep = torch.tensor([True, True, False, True, True, True, True, True])
    dropped, _ = _update(monkeypatch, s, advantage=Advantage(name="drop", fn=lambda b, st: (adv, keep)))
    # The same update as a batch where completion 2 has no counted tokens at all.
    mask = FIXED_BATCH["completion_mask"].clone()
    mask[2] = 0
    monkeypatch.setitem(FIXED_BATCH, "completion_mask", mask)
    want, _ = _update(monkeypatch, s)
    for name in want:
        torch.testing.assert_close(dropped[name], want[name])


def test_diagnostics(monkeypatch):
    # Group 1 [1, 0, 0, 1]: variance 0.25. Group 2 [0, 1, 1, 1]: 0.1875. No all-equal group.
    _, stats = _update(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8))
    assert stats["zero_adv_frac"] == 0.0 and stats["reward_var"] == pytest.approx((0.25 + 0.1875) / 2)
    assert stats["rollouts"] == 8 and stats["grad_norm"] > 0 and stats["entropy"] > 0
    _, stats = _update(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8), rewards=[1.0] * 4 + [0, 1, 0, 1])
    assert stats["zero_adv_frac"] == 0.5, "one of the two groups is all-correct, so it carries no gradient"
    _, stats = _update(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8), rewards=[1, 0, 0.5, 0.5] + [1.0] * 4)
    assert stats["zero_adv_frac"] == 0.5, "a group counts only when ALL its advantages are 0, not some"


def test_truncated_completions_can_be_scored_wrong(monkeypatch):
    def with_truncation(model, tokenizer, prompts, group_size, max_new_tokens, temperature):
        b = {k: v.clone() for k, v in FIXED_BATCH.items()}
        b["truncated"] = ~(b["completion_ids"] == EOS).any(dim=1)    # rows 1, 3, 5, 7 never end
        return b
    monkeypatch.setattr(src.train.loop, "sample_group", with_truncation)
    seen = {}

    def spy(batch, state):
        seen["rewards"] = batch.rewards.tolist()
        return torch.zeros_like(batch.rewards)

    model = make_model()
    for flag, want in ((False, [1.0] * 8), (True, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])):
        stats = grpo_step(model, torch.optim.SGD(model.parameters(), lr=0.0), make_tokenizer(), ["p0", "p1"],
                          ["a0", "a1"], lambda t, a: [1.0] * 8,
                          GRPOSettings(group_size=4, max_new_tokens=4, truncated_is_wrong=flag),
                          advantage=Advantage(name="spy", fn=spy))
        assert seen["rewards"] == want
    assert stats["truncated_frac"] == 0.5


def test_uniform_groups_and_advantage_scale(monkeypatch):
    # Sign gives an all-correct group +1 everywhere: the group is uniform but its advantages are not 0.
    _, stats = _update(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8), rewards=[1.0] * 4 + [0, 1, 0, 1],
                       advantage=build_advantage("sign") if _has_sign() else None)
    assert stats["uniform_frac"] == 0.5
    _, stats = _update(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8, scale="none"),
                       rewards=[1.0, 0, 0, 0] + [1.0] * 4)
    assert stats["adv_abs_mean"] == pytest.approx((0.75 + 3 * 0.25) / 8)


def _has_sign():
    import importlib.util
    return importlib.util.find_spec("src.advantages.sign") is not None


def test_mc_grpo_through_the_step(monkeypatch):
    """MC-GRPO end to end: G + 1 sampled, the median completion of each group dropped, and the update
    the same as a batch that never had those rows."""
    seen = {}
    _fixed(monkeypatch, seen)
    method = build_advantage("mc_grpo")
    # FIXED_BATCH has 2 prompts x 4 rows, so group size 3 plus the 1 extra rollout.
    # FIXED_REWARDS: group 1 [1, 0, 0, 1] -> median (lower middle) 0, row 1 dropped;
    #                group 2 [0, 1, 1, 1] -> median 1, row 5 dropped.
    s = GRPOSettings(group_size=3, max_new_tokens=8, loss_type="grpo")
    model = make_model(logit_scale=5.0)
    before = copy.deepcopy(model.state_dict())
    stats = grpo_step(model, torch.optim.SGD(model.parameters(), lr=1.0), make_tokenizer(), ["p0", "p1"],
                      ["a0", "a1"], lambda t, a: FIXED_REWARDS, s, advantage=method)
    assert seen["group_size"] == 4 and stats["rollouts"] == 8
    got = {n: before[n] - v for n, v in model.state_dict().items()}
    adv, keep = method(src.advantages.base.AdvantageBatch(
        rewards=torch.tensor(FIXED_REWARDS), group_size=4, prompt_ids=torch.arange(2),
        lengths=FIXED_BATCH["completion_mask"].sum(dim=1)))
    assert keep.tolist() == [True, False, True, True, True, False, True, True]
    mask = FIXED_BATCH["completion_mask"].clone() * keep.unsqueeze(1)
    monkeypatch.setitem(FIXED_BATCH, "completion_mask", mask)
    fixed_adv = Advantage(name="fixed", fn=lambda b, st: adv)
    want, _ = _update(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8, loss_type="grpo"), advantage=fixed_adv)
    for name in want:
        torch.testing.assert_close(got[name], want[name])


# ---- DAPO's dynamic sampling --------------------------------------------------------------------

REFILL_BATCH = {   # one prompt x 4, narrower than FIXED_BATCH on both sides, to test the padding
    "prompt_ids": torch.tensor([[17, 18]] * 4),
    "prompt_mask": torch.tensor([[1, 1]] * 4),
    "completion_ids": torch.tensor([[42, EOS, PAD], [43, 44, 45], [46, 47, EOS], [EOS, PAD, PAD]]),
    "completion_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 0, 0]]),
}


def _by_answer(texts, answers):
    """'u' answers: every completion right (a uniform group). Others: right, wrong, right, wrong."""
    return [1.0 if a == "u" else float(i % 2 == 0) for i, a in enumerate(answers)]


def _dynamic(monkeypatch, settings, answers, refill_answer="m", max_rollouts=None, lr=1.0):
    calls = []

    def sampler(model, tokenizer, prompts, group_size, max_new_tokens, temperature):
        calls.append(list(prompts))
        n = len(prompts) * group_size
        if len(calls) == 1:
            return {k: v[:n].clone() for k, v in FIXED_BATCH.items()}
        return {k: v.repeat(len(prompts), 1)[:n].clone() for k, v in REFILL_BATCH.items()}   # REFILL_BATCH per prompt

    monkeypatch.setattr(src.train.loop, "sample_group", sampler)
    seen = {}

    def spy(batch, state):
        seen["batch"] = batch
        return src.advantages.grpo.group_advantages(batch.rewards, batch.group_size)

    refills = iter(range(100, 200))

    def refill(n):
        return [f"r{k}" for k in range(n)], [refill_answer] * n, [next(refills) for _ in range(n)]

    model = make_model(logit_scale=5.0)
    stats = grpo_step(model, torch.optim.SGD(model.parameters(), lr=lr), make_tokenizer(), ["p0", "p1"], answers,
                      _by_answer, settings, advantage=Advantage(name="spy", fn=spy), prompt_ids=[7, 8],
                      refill=refill, max_rollouts=max_rollouts)
    return stats, seen.get("batch"), calls


def test_dynamic_sampling_replaces_uniform_groups_and_counts_every_rollout(monkeypatch):
    s = GRPOSettings(group_size=4, max_new_tokens=8, drop_uniform_groups=True)
    stats, batch, calls = _dynamic(monkeypatch, s, answers=["m", "u"])
    assert calls == [["p0", "p1"], ["r0"]], "prompt p1's group was all right, so one fresh prompt is sampled"
    assert stats["rollouts"] == 12, "8 + 4: the dropped group's rollouts still count"
    assert (stats["groups_kept"], stats["groups_dropped"], stats["refills"]) == (2, 1, 1)
    assert batch.prompt_ids.tolist() == [7, 100] and batch.rewards.tolist() == [1, 0, 1, 0] * 2


def test_dynamic_sampling_stops_at_the_rollout_budget(monkeypatch):
    s = GRPOSettings(group_size=4, max_new_tokens=8, drop_uniform_groups=True)
    stats, batch, calls = _dynamic(monkeypatch, s, answers=["m", "u"], max_rollouts=8)
    assert len(calls) == 1 and stats["rollouts"] == 8 and stats["groups_kept"] == 1
    assert batch.prompt_ids.tolist() == [7], "the step trains on the one mixed group it has"


def test_dynamic_sampling_gives_up_after_max_refills(monkeypatch):
    s = GRPOSettings(group_size=4, max_new_tokens=8, drop_uniform_groups=True, max_refills=3)
    stats, batch, calls = _dynamic(monkeypatch, s, answers=["u", "u"], refill_answer="u")
    assert stats.get("skipped") and batch is None and len(calls) == 4
    assert stats["rollouts"] == 8 + 3 * 2 * 4, "each refill asks for the 2 missing groups"


def test_without_dynamic_sampling_uniform_groups_stay(monkeypatch):
    stats, batch, calls = _dynamic(monkeypatch, GRPOSettings(group_size=4, max_new_tokens=8), answers=["m", "u"])
    assert len(calls) == 1 and stats["rollouts"] == 8 and "groups_kept" not in stats
    assert batch.prompt_ids.tolist() == [7, 8]


def test_batches_of_different_widths_are_padded_on_the_right_sides():
    a = {k: v[:4] for k, v in FIXED_BATCH.items()}
    a["truncated"] = torch.tensor([False, True, False, True])
    b = dict(REFILL_BATCH, truncated=torch.tensor([False, False, False, False]))
    out = src.train.loop._cat_batches(a, b, PAD)
    assert out["prompt_ids"].shape == (8, 3) and out["completion_ids"].shape == (8, 4)
    assert out["prompt_ids"][4].tolist() == [PAD, 17, 18] and out["prompt_mask"][4].tolist() == [0, 1, 1]
    assert out["completion_ids"][5].tolist() == [43, 44, 45, PAD] and out["completion_mask"][5].tolist() == [1, 1, 1, 0]
    assert out["truncated"].tolist() == [False, True, False, True, False, False, False, False]
    torch.testing.assert_close(out["completion_mask"][:4], a["completion_mask"])


def test_padding_a_refill_does_not_change_its_log_probs():
    """Left-padding a prompt and right-padding a completion must leave the counted log-probs unchanged."""
    model = make_model(logit_scale=5.0)
    out = src.train.loop._cat_batches({k: v[:4] for k, v in FIXED_BATCH.items()}, REFILL_BATCH, PAD)
    ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)[4:]
    am = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)[4:]
    padded = per_token_logps(model, ids, am, 4, 1.0)[:, :3] * REFILL_BATCH["completion_mask"]
    ids0 = torch.cat([REFILL_BATCH["prompt_ids"], REFILL_BATCH["completion_ids"]], dim=1)
    am0 = torch.cat([REFILL_BATCH["prompt_mask"], REFILL_BATCH["completion_mask"]], dim=1)
    plain = per_token_logps(model, ids0, am0, 3, 1.0) * REFILL_BATCH["completion_mask"]
    torch.testing.assert_close(padded, plain, rtol=1e-5, atol=1e-5)
