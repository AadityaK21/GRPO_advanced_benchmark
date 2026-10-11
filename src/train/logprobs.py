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
    raise NotImplementedError
