"""Per-token log-probabilities of the completion under a model."""
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

# Completion tokens scored per piece. Each piece builds a [CHUNK, vocab] block of logits: with Qwen's
# 152k vocabulary in fp32 that is about 150 MB, against 2.5 GB for 8 completions of 512 tokens at once.
CHUNK_SIZE = 256


def per_token_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int,
                    temperature: float = 1.0, chunk_size: int | None = CHUNK_SIZE) -> torch.Tensor:
    """Log-probability of each of the last `logits_to_keep` tokens in each row, given the tokens before it.

    input_ids, attention_mask: [B, P + C], the left-padded prompt followed by the completion.
    Returns [B, logits_to_keep]: entry [b, j] is log p(completion token j | everything before it).

    The logits are divided by `temperature` before the softmax, so these are the log-probs of the
    distribution you sampled from. The graph is kept: the loss backpropagates through this.

    chunk_size: score this many completion tokens at a time (see per_token_logps_and_entropy).
    None builds all the logits at once, which is the simple version and needs far more memory.
    """
    return per_token_logps_and_entropy(model, input_ids, attention_mask, logits_to_keep, temperature,
                                       chunk_size, entropy=False)[0]


def per_token_logps_and_entropy(model, input_ids, attention_mask, logits_to_keep: int, temperature: float = 1.0,
                                chunk_size: int | None = CHUNK_SIZE, entropy: bool = True):
    """per_token_logps, plus the entropy of each position's sampling distribution (a diagnostic, no graph).

    Returns (logps [B, C], entropy [B, C] or None).

    Memory. The model body gives one hidden vector per position (896 numbers for Qwen2.5-0.5B). The
    output layer turns each into 152k logits, and that step is what used 11.7 GB: logits for every
    completion token at once, plus copies for the softmax and its gradient. Here the hidden vectors go
    through the output layer `chunk_size` tokens at a time. With gradients on, each piece is
    checkpointed: its logits are freed straight away and rebuilt in the backward pass, so at most one
    piece of logits exists at any moment.

    Models without get_decoder()/get_output_embeddings() fall back to building all the logits at once.
    """
    k = logits_to_keep
    parts = _body_and_head(model) if chunk_size else None
    if parts is None:
        # One forward for the last k + 1 positions only, as TRL does: the prompt's other logits are
        # never used, and on Qwen each position costs a 152k-wide row.
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       logits_to_keep=k + 1, use_cache=False).logits     # no KV cache when scoring
        # The one-token shift: logits at position t predict token t + 1, so drop the last kept
        # position (it predicts past the end).
        return _score(logits[:, :-1, :], input_ids[:, -k:], temperature, entropy)

    body, head = parts
    out = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]   # [B, P + C, H]
    # The same one-token shift, on hidden vectors: positions -(k+1) .. -2 predict the k completion tokens.
    hidden = hidden[:, -(k + 1):-1, :]
    B = hidden.size(0)
    flat_h = hidden.reshape(B * k, hidden.size(-1))
    flat_y = input_ids[:, -k:].reshape(B * k)

    score = partial(_score_hidden, head, temperature=temperature, entropy=entropy)
    logps, ents = [], []
    for start in range(0, B * k, chunk_size):
        h, y = flat_h[start:start + chunk_size], flat_y[start:start + chunk_size]
        if torch.is_grad_enabled():
            lp, ent = checkpoint(score, h, y, use_reentrant=False)
        else:
            lp, ent = score(h, y)
        logps.append(lp)
        ents.append(ent)
    logps = torch.cat(logps).view(B, k)
    return logps, (torch.cat(ents).view(B, k) if entropy else None)


def _body_and_head(model):
    """(model body returning hidden states, output layer), or None when the model does not expose them."""
    get_body = getattr(model, "get_decoder", None)
    get_head = getattr(model, "get_output_embeddings", None)
    if not callable(get_body) or not callable(get_head):
        return None
    body, head = get_body(), get_head()
    if body is None or head is None:
        return None
    return body, head


def _score_hidden(head, h, y, temperature, entropy):
    return _score(head(h), y, temperature, entropy)


def _score(logits, targets, temperature, entropy):
    """Log-prob of each target under softmax(logits / temperature), and optionally each row's entropy.

    log softmax(x)[y] = x[y] - logsumexp(x): pick one logit per position instead of building the
    full log-softmax. Works on [..., V] logits with [...] targets.
    """
    logits = logits.float()
    if temperature != 1.0:
        logits = logits / temperature                    # temperature BEFORE the softmax
    lse = torch.logsumexp(logits, dim=-1)
    logps = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) - lse
    ent = None
    if entropy:
        with torch.no_grad():
            # H = -sum p log p = logsumexp(x) - sum p x
            ent = lse.detach() - (torch.softmax(logits, dim=-1) * logits).sum(dim=-1)
    return logps, ent
