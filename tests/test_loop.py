"""Checks for src/train/loop.py: one GRPO step, start to finish."""
import copy

import pytest
import torch

import src.train.loop
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
