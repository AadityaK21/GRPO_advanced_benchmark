"""One GRPO training step, start to finish. Your training script calls this in a loop."""
from dataclasses import dataclass


@dataclass
class GRPOSettings:
    group_size: int = 8          # completions per prompt
    max_new_tokens: int = 256    # also the max_completion_length for loss_type="dr_grpo"
    temperature: float = 1.0
    num_updates: int = 1         # optimizer updates per sampled batch; TRL calls this num_iterations
    loss_type: str = "grpo"
    eps_low: float = 0.2
    eps_high: float = 0.2
    beta: float = 0.0            # KL coefficient; needs ref_model when not 0
    scale: str = "group"         # advantage scaling, see group_advantages
    max_grad_norm: float | None = None   # if set, clip the gradient norm before each update (TRL: 1.0)


def grpo_step(model, optimizer, tokenizer, prompts: list[str], answers: list[str], reward_fn,
              settings: GRPOSettings, ref_model=None) -> dict:
    """Sample, score, compute advantages, then make settings.num_updates updates on the GRPO loss.

    Use your own pieces: sample_group, completion_mask, group_advantages, per_token_logps, grpo_loss.
    - Score with reward_fn(texts, completion_answers), both lists passed positionally; it returns one
      float per completion. texts are the decoded completions (only the tokens that count), and
      completion_answers[k] is the answer of the prompt that completion k belongs to.
    - Old log-probs are those of the model that sampled the batch. Reference log-probs come from
      ref_model. Neither needs gradients.

    Returns a dict with at least:
        "reward":    mean reward of the sampled completions
        "clip_frac": fraction of counted tokens clipped in the last update. A token is clipped when
                     its ratio is above 1 + eps_high with a positive advantage, or below 1 - eps_low
                     with a negative one.
        "kl":        mean per-token k3 KL to ref_model over counted tokens in the last update
                     (0.0 when beta is 0)
    """
    raise NotImplementedError
