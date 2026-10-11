"""Checks for src/train/logprobs.py."""
import pytest
import torch

from tiny import PAD, make_model
from src.train.logprobs import per_token_logps


def _batch():
    """Two prompts, the first left-padded by 2, then 4 completion tokens each."""
    prompts = torch.tensor([[PAD, PAD, 7, 8, 9], [3, 4, 5, 6, 7]])
    prompt_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    completions = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    input_ids = torch.cat([prompts, completions], dim=1)
    attention_mask = torch.cat([prompt_mask, torch.ones_like(completions)], dim=1)
    return input_ids, attention_mask, completions.size(1)


def _brute_force(model, input_ids, attention_mask, k, temperature):
    """Score each completion token on its own, by running the model on the unpadded prefix before it."""
    out = torch.zeros(input_ids.size(0), k)
    with torch.no_grad():
        for b in range(input_ids.size(0)):
            row = input_ids[b][attention_mask[b].bool()]
            for j in range(k):
                pos = row.size(0) - k + j
                logits = model(input_ids=row[:pos].unsqueeze(0)).logits[0, -1] / temperature
                out[b, j] = torch.log_softmax(logits, dim=-1)[row[pos]]
    return out


@pytest.mark.parametrize("temperature", [1.0, 0.5])
def test_matches_scoring_each_token_on_its_own(temperature):
    model = make_model(logit_scale=10.0)
    input_ids, attention_mask, k = _batch()
    got = per_token_logps(model, input_ids, attention_mask, k, temperature)
    assert got.shape == (2, k), "one log-prob per completion token: shape [B, logits_to_keep]"
    want = _brute_force(model, input_ids, attention_mask, k, temperature)
    torch.testing.assert_close(
        got.detach(), want, rtol=1e-4, atol=1e-4,
        msg="logits at position t predict token t+1; divide the logits by the temperature before the softmax",
    )


def test_keeps_the_graph_for_the_loss():
    model = make_model()
    input_ids, attention_mask, k = _batch()
    per_token_logps(model, input_ids, attention_mask, k).sum().backward()
    grad = model.lm_head.weight.grad
    assert grad is not None and grad.abs().sum() > 0, "no torch.no_grad() or .detach() inside this function"
