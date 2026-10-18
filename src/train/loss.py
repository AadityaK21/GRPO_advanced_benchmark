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
    if loss_type not in ("grpo", "bnpo", "dapo", "dr_grpo"):
        raise ValueError(f"unknown loss_type {loss_type!r}; use grpo, bnpo, dapo or dr_grpo")
    mask = mask.to(logps.dtype)
    # Constants: the sampling policy, the advantages and the reference get no gradient.
    old = logps.detach() if old_logps is None else old_logps.detach()
    adv = advantages.detach().to(logps.dtype).unsqueeze(1)          # [B, 1], one per completion

    # PPO's clipped surrogate, per token. The ratio is exp of a difference of log-probs.
    ratio = torch.exp(logps - old)
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)        # separate bounds: DAPO's clip-higher
    per_token = -torch.min(ratio * adv, clipped * adv)               # min, then minus: we minimise

    if beta != 0.0:
        if ref_logps is None:
            raise ValueError("beta != 0 needs ref_logps")
        # k3 estimator of KL(policy || reference): always >= 0, and 0 when they agree.
        d = ref_logps.detach() - logps
        per_token = per_token + beta * (torch.exp(d) - d - 1)

    # The token averaging: the one place GRPO, DAPO and Dr. GRPO differ in the loss.
    if loss_type == "grpo":
        # Mean over each completion's own tokens, then over completions: tokens of short
        # completions weigh more. This is the length bias Dr. GRPO points out.
        per_completion = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return per_completion.mean()
    if loss_type in ("bnpo", "dapo"):
        # Every counted token weighs the same.
        return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
    # dr_grpo: every token weighs the same, over a fixed denominator that does not depend on
    # how long this batch's answers happen to be.
    if max_completion_length is None:
        raise ValueError("loss_type='dr_grpo' needs max_completion_length")
    return (per_token * mask).sum() / (per_token.size(0) * max_completion_length)
