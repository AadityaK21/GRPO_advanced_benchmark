"""Per-token log-probabilities of the completion under a model."""
import torch


def per_token_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int,
                    temperature: float = 1.0) -> torch.Tensor:
    """Log-probability of each of the last `logits_to_keep` tokens in each row, given the tokens before it.

    input_ids, attention_mask: [B, P + C], the left-padded prompt followed by the completion.
    Returns [B, logits_to_keep]: entry [b, j] is log p(completion token j | everything before it).

    Divide the logits by `temperature` before the softmax, so these are the log-probs of the
    distribution you sampled from. Keep the graph: the loss backpropagates through this.

    Memory: Qwen's vocabulary has about 152k tokens, so a full log_softmax over [B, C, vocab]
    is large. You only need one value per position.
    """
    # Ask the model for the last logits_to_keep + 1 positions only, as TRL does: the prompt's
    # other logits are never used, and on Qwen each position costs a 152k-wide row.
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   logits_to_keep=logits_to_keep + 1).logits
    # The one-token shift: logits at position t predict token t + 1. So the logits for the
    # completion tokens are the kept positions except the last one (which predicts past the end).
    logits = logits[:, :-1, :].float() / temperature            # [B, C, V]; temperature BEFORE softmax
    targets = input_ids[:, -logits_to_keep:]                     # [B, C], the completion tokens
    # log softmax(x)[y] = x[y] - logsumexp(x). Pick one logit per position instead of building
    # the full [B, C, V] log-softmax.
    picked = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return picked - torch.logsumexp(logits, dim=-1)
