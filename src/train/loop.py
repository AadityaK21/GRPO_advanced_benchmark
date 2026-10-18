"""One GRPO training step, start to finish. Your training script calls this in a loop."""
from dataclasses import dataclass

import torch

from src.advantages.grpo import group_advantages
from src.train.logprobs import per_token_logps
from src.train.loss import grpo_loss
from src.train.sample import sample_group


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
    st = settings
    if st.beta != 0.0 and ref_model is None:
        raise ValueError("settings.beta != 0 needs a ref_model")

    # 1. Sample G completions per prompt from the current model.
    batch = sample_group(model, tokenizer, prompts, st.group_size, st.max_new_tokens, st.temperature)
    completion_ids, cmask = batch["completion_ids"], batch["completion_mask"]

    # 2. Score them. Decode only the counted tokens; each completion is paired with the answer of
    #    its own prompt (answers repeated G times in a row, matching the row layout).
    texts = [tokenizer.decode(ids[: int(m.sum())], skip_special_tokens=True)
             for ids, m in zip(completion_ids, cmask)]
    completion_answers = [a for a in answers for _ in range(st.group_size)]
    rewards = torch.as_tensor(reward_fn(texts, completion_answers), dtype=torch.float32)

    # 3. Advantages from the group statistics. No critic.
    advantages = group_advantages(rewards, st.group_size, st.scale).to(completion_ids.device)

    # 4. Old and reference log-probs, once, before any update. Both are constants.
    input_ids = torch.cat([batch["prompt_ids"], completion_ids], dim=1)
    attention_mask = torch.cat([batch["prompt_mask"], cmask], dim=1)
    C = completion_ids.size(1)
    with torch.no_grad():
        # With one update the sampling model IS the current model, so old = logps.detach()
        # (pass None, as TRL does). With several, freeze the sampling model's log-probs now:
        # recomputing them every update would keep the ratio at 1 and clipping would never act.
        old_logps = (per_token_logps(model, input_ids, attention_mask, C, st.temperature)
                     if st.num_updates > 1 else None)
        ref_logps = (per_token_logps(ref_model, input_ids, attention_mask, C, st.temperature)
                     if st.beta != 0.0 else None)

    # 5. Several updates on the same batch.
    for _ in range(st.num_updates):
        logps = per_token_logps(model, input_ids, attention_mask, C, st.temperature)
        loss = grpo_loss(logps, old_logps, advantages, cmask, ref_logps=ref_logps, beta=st.beta,
                         eps_low=st.eps_low, eps_high=st.eps_high, loss_type=st.loss_type,
                         max_completion_length=st.max_new_tokens)
        optimizer.zero_grad()
        loss.backward()
        if st.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), st.max_grad_norm)
        optimizer.step()

    # 6. Statistics from the last update, over counted tokens only.
    with torch.no_grad():
        lp = logps.detach()
        ratio = torch.exp(lp - (lp if old_logps is None else old_logps))
        a = advantages.unsqueeze(1)
        clipped = ((ratio > 1 + st.eps_high) & (a > 0)) | ((ratio < 1 - st.eps_low) & (a < 0))
        m = cmask.float()
        n_tokens = m.sum().clamp(min=1.0)
        clip_frac = ((clipped.float() * m).sum() / n_tokens).item()
        kl = 0.0
        if ref_logps is not None:
            d = ref_logps - lp
            kl = (((torch.exp(d) - d - 1) * m).sum() / n_tokens).item()
    return {"reward": rewards.mean().item(), "clip_frac": clip_frac, "kl": kl, "loss": loss.item(),
            "completion_length": m.sum(dim=1).mean().item()}
