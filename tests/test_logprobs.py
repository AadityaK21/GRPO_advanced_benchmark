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


# ---- Chunked scoring (the memory fix) -------------------------------------------------------------

def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 1000])   # 3 leaves a ragged last piece of 2 of the 8 tokens
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_chunked_matches_all_at_once_values_and_gradients(chunk_size, temperature):
    input_ids, attention_mask, k = _batch()
    weights = torch.linspace(-1.0, 1.0, 2 * k).view(2, k)   # any fixed weighting, so every position matters

    full_model = make_model(logit_scale=5.0).train()
    full = per_token_logps(full_model, input_ids, attention_mask, k, temperature, chunk_size=None)
    (full * weights).sum().backward()

    chunk_model = make_model(logit_scale=5.0).train()
    chunked = per_token_logps(chunk_model, input_ids, attention_mask, k, temperature, chunk_size=chunk_size)
    (chunked * weights).sum().backward()

    torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-5, msg="chunking must not change the log-probs")
    g_full, g_chunk = _grads(full_model), _grads(chunk_model)
    assert g_full.keys() == g_chunk.keys()
    for name in g_full:
        torch.testing.assert_close(g_chunk[name], g_full[name], rtol=1e-4, atol=1e-6,
                                   msg=f"chunking must not change the gradient of {name}")


def test_chunked_works_with_gradient_checkpointing():
    input_ids, attention_mask, k = _batch()
    plain = make_model().train()
    per_token_logps(plain, input_ids, attention_mask, k, chunk_size=None).sum().backward()

    ckpt = make_model().train()
    ckpt.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    per_token_logps(ckpt, input_ids, attention_mask, k, chunk_size=3).sum().backward()
    for name, g in _grads(plain).items():
        torch.testing.assert_close(_grads(ckpt)[name], g, rtol=1e-4, atol=1e-6)


def test_models_without_a_body_fall_back_to_the_full_path():
    class Wrapper(torch.nn.Module):          # like AdapterOff without its extra methods
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, **kwargs):
            return self.inner(**kwargs)

    model = make_model()
    input_ids, attention_mask, k = _batch()
    want = per_token_logps(model, input_ids, attention_mask, k, chunk_size=None)
    got = per_token_logps(Wrapper(model), input_ids, attention_mask, k, chunk_size=3)
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_entropy_matches_the_definition(chunk_size):
    from src.train.logprobs import per_token_logps_and_entropy
    model = make_model(logit_scale=5.0)
    input_ids, attention_mask, k = _batch()
    temperature = 0.7
    logps, ent = per_token_logps_and_entropy(model, input_ids, attention_mask, k, temperature, chunk_size)
    assert not ent.requires_grad, "entropy is a logged diagnostic: no graph"

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -k - 1:-1] / temperature
        p = torch.softmax(logits, dim=-1)
        want = -(p * torch.log(p)).sum(dim=-1)
    torch.testing.assert_close(ent, want, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(logps, per_token_logps(model, input_ids, attention_mask, k, temperature, None))
