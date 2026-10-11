"""Checks for src/train/sample.py."""
import pytest
import torch

from src.train.sample import completion_mask, sample_group
from tiny import EOS, PAD, VOCAB, make_model, make_tokenizer, tid


def _first_eos_mask(ids, eos_ids):
    mask = torch.zeros_like(ids)
    for b in range(ids.size(0)):
        for t in range(ids.size(1)):
            mask[b, t] = 1
            if int(ids[b, t]) in eos_ids:
                break
    return mask


# ------------------------------------------------------- the completion mask


def test_mask_keeps_tokens_up_to_and_including_the_first_eos():
    ids = torch.tensor([[5, EOS, 7, EOS], [5, 6, 7, 8], [EOS, 9, 9, 9]])
    want = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    assert torch.equal(completion_mask(ids, EOS).long(), want), "the EOS token is trained on; what follows it is not"


def test_mask_accepts_several_eos_ids():
    ids = torch.tensor([[5, 9, 7, EOS], [5, 6, EOS, 9]])
    want = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    assert torch.equal(completion_mask(ids, [EOS, 9]).long(), want), "Qwen2.5-Instruct stops on either of two EOS ids"


# ------------------------------------------------------------- sampling


def test_layout_and_left_padding():
    model, tokenizer = make_model(), make_tokenizer()
    torch.manual_seed(0)
    out = sample_group(model, tokenizer, ["t5 t6 t7", "t8"], group_size=3, max_new_tokens=5, temperature=1.0)
    p, pm, c, cm = out["prompt_ids"], out["prompt_mask"], out["completion_ids"], out["completion_mask"]
    assert p.shape == (6, 3) and pm.shape == (6, 3), "N * G rows, prompts padded to the longest"
    assert torch.equal(p[:3], torch.tensor([[tid(5), tid(6), tid(7)]] * 3)), "rows 0..G-1 are prompt 0"
    assert torch.equal(pm[3:], torch.tensor([[0, 0, 1]] * 3)), "pad on the left, so every completion starts right after its prompt"
    assert torch.equal(p[3:, 2], torch.tensor([tid(8)] * 3)) and torch.all(p[3:, :2] == PAD)
    assert c.size(0) == 6 and c.size(1) <= 5 and cm.shape == c.shape
    assert torch.equal(cm.long(), _first_eos_mask(c, {EOS})), "completion_mask: up to and including the first EOS"


def test_sampling_is_reproducible_under_a_seed():
    model, tokenizer = make_model(), make_tokenizer()
    runs = []
    for _ in range(2):
        torch.manual_seed(3)
        runs.append(sample_group(model, tokenizer, ["t1 t2", "t3"], group_size=4, max_new_tokens=6, temperature=1.0))
    assert torch.equal(runs[0]["completion_ids"], runs[1]["completion_ids"])


@pytest.mark.parametrize("logit_scale", [1.0, 10.0])
def test_samples_come_from_softmax_of_the_logits_over_temperature(logit_scale):
    model, tokenizer = make_model(logit_scale=logit_scale), make_tokenizer()
    # A checkpoint's generation_config.json can set its own sampling rules, and generate() uses them unless
    # you override them. Qwen2.5-0.5B-Instruct's sets temperature 0.7, top_k 20, top_p 0.8 and
    # repetition_penalty 1.1. These values are more extreme so a test can see them.
    cfg = model.generation_config
    cfg.do_sample, cfg.temperature, cfg.top_k, cfg.top_p, cfg.repetition_penalty = False, 1.5, 3, 0.5, 2.0
    prompt = " ".join(f"t{k}" for k in range(3, 23))
    torch.manual_seed(0)
    out = sample_group(model, tokenizer, [prompt], group_size=4000, max_new_tokens=1, temperature=0.7)
    first = out["completion_ids"][:, 0]
    seen = torch.bincount(first, minlength=VOCAB).double() / first.numel()
    with torch.no_grad():
        ids = tokenizer([prompt], return_tensors="pt")["input_ids"]
        want = torch.softmax(model(input_ids=ids).logits[0, -1].double() / 0.7, dim=-1)
    distance = 0.5 * (seen - want).abs().sum().item()
    assert distance < 0.08, (
        f"your samples are {distance:.2f} (total variation) from softmax(logits / temperature). Sample with "
        "do_sample=True and your temperature, and switch off top_k (0), top_p (1.0) and repetition_penalty (1.0). "
        "GRPO's log-probs assume this exact distribution."
    )
