"""The clipped GRPO loss, the KL term, and the token averaging."""
import torch


def grpo_loss(logps: torch.Tensor, old_logps: torch.Tensor | None, advantages: torch.Tensor,
              mask: torch.Tensor, *, ref_logps: torch.Tensor | None = None, beta: float = 0.0,
              eps_low: float = 0.2, eps_high: float = 0.2, loss_type: str = "grpo",
              max_completion_length: int | None = None) -> torch.Tensor:
    """The GRPO loss to minimise, as a scalar.

    logps:      [B, C] log-probs under the current model, with gradient.
    old_logps:  [B, C] log-probs under the model that sampled the batch, or None for the first
                update, which means logps.detach().
    advantages: [B], one per completion.
    mask:       [B, C], 1 for the completion tokens that count.

    Per token: -min(r * A, clip(r, 1 - eps_low, 1 + eps_high) * A), where r = exp(logps - old_logps).
    If beta != 0, add beta * (exp(ref - logps) - (ref - logps) - 1), the k3 estimate of the KL
    to the reference model, where ref = ref_logps.

    Then average over the counted tokens, according to loss_type:
        "grpo":          mean over each completion's tokens, then mean over completions
        "bnpo", "dapo":  sum over all tokens / number of tokens
        "dr_grpo":       sum over all tokens / (B * max_completion_length)
    (TRL's "dapo" divides by the token count of the whole generation batch. With one batch and
    no gradient accumulation, that is the same as "bnpo".)

    Old log-probs, reference log-probs and advantages are constants: no gradient flows into them.
    An unknown loss_type raises ValueError.
    """
    raise NotImplementedError
