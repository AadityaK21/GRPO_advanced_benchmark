"""One GRPO training step, start to finish. Your training script calls this in a loop.

Nothing method-specific lives here: the advantage comes from an Advantage object (src/advantages/),
and a method's loss changes come in through GRPOSettings.
"""
from dataclasses import dataclass

import torch

from src.advantages.base import Advantage, AdvantageBatch
from src.advantages.grpo import group_advantages
from src.train.logprobs import per_token_logps, per_token_logps_and_entropy
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
    scale: str = "group"         # advantage scaling when no Advantage object is given, see group_advantages
    max_grad_norm: float | None = None   # if set, clip the gradient norm before each update (TRL: 1.0)
    micro_batch_size: int | None = None  # completions per forward/backward pass. Memory only: the
                                         # gradients are summed so the update is the full batch's


def grpo_step(model, optimizer, tokenizer, prompts: list[str], answers: list[str], reward_fn,
              settings: GRPOSettings, ref_model=None, advantage: Advantage | None = None,
              prompt_ids: list[int] | None = None, step: int = 0) -> dict:
    """Sample, score, compute advantages, then make settings.num_updates updates on the GRPO loss.

    - reward_fn(texts, completion_answers), both lists passed positionally, returns one float per
      completion. texts are the decoded completions (only the tokens that count), and
      completion_answers[k] is the answer of the prompt that completion k belongs to.
    - advantage: the method. None means vanilla GRPO with settings.scale.
    - prompt_ids: each prompt's index in the training set, for methods that track prompts across
      steps. Defaults to 0 .. len(prompts) - 1.

    Returns a dict of statistics:
        reward, clip_frac, kl, loss, completion_length, truncated_frac   (as before)
        zero_adv_frac:  fraction of groups whose advantages are all 0, so they give no gradient
                        (2605.21125's "advantage collapse rate")
        reward_var:     mean within-group reward variance (divides by G)
        entropy:        mean per-token entropy of the sampling distribution over counted tokens
        grad_norm:      gradient norm before clipping, last update
        rollouts:       completions sampled this step (counts any extra rollouts)
    clip_frac, kl, entropy and grad_norm are from the last update. A token is clipped when its
    ratio is above 1 + eps_high with a positive advantage, or below 1 - eps_low with a negative one.
    """
    st = settings
    if st.beta != 0.0 and ref_model is None:
        raise ValueError("settings.beta != 0 needs a ref_model")
    G = st.group_size + (advantage.extra_rollouts if advantage is not None else 0)

    # 1. Sample G completions per prompt from the current model.
    batch = sample_group(model, tokenizer, prompts, G, st.max_new_tokens, st.temperature)
    completion_ids, cmask = batch["completion_ids"], batch["completion_mask"]
    device = completion_ids.device

    # 2. Score them. Decode only the counted tokens; each completion is paired with the answer of
    #    its own prompt (answers repeated G times in a row, matching the row layout).
    texts = [tokenizer.decode(ids[: int(m.sum())], skip_special_tokens=True)
             for ids, m in zip(completion_ids, cmask)]
    completion_answers = [a for a in answers for _ in range(G)]
    rewards = torch.as_tensor(reward_fn(texts, completion_answers), dtype=torch.float32)

    input_ids = torch.cat([batch["prompt_ids"], completion_ids], dim=1)
    attention_mask = torch.cat([batch["prompt_mask"], cmask], dim=1)
    C = completion_ids.size(1)
    N = completion_ids.size(0)
    mb = st.micro_batch_size or N
    pieces = [slice(i, min(i + mb, N)) for i in range(0, N, mb)]

    def score_all(m):
        """Log-probs for the whole batch, a micro-batch at a time, with gradients off."""
        with torch.no_grad():
            return torch.cat([per_token_logps(m, input_ids[s], attention_mask[s], C, st.temperature)
                              for s in pieces])

    # 3. Old and reference log-probs, once, before any update. Both are constants.
    # With one update the sampling model IS the current model, so old = logps.detach()
    # (pass None, as TRL does). With several, freeze the sampling model's log-probs now:
    # recomputing them every update would keep the ratio at 1 and clipping would never act.
    needs_logps = advantage is not None and "seq_logps" in advantage.needs
    old_logps = score_all(model) if (st.num_updates > 1 or needs_logps) else None
    ref_logps = score_all(ref_model) if st.beta != 0.0 else None

    # 4. Advantages. No critic.
    if advantage is None:
        advantages, keep = group_advantages(rewards, G, st.scale), None
    else:
        ids = list(range(len(prompts))) if prompt_ids is None else list(prompt_ids)
        adv_batch = AdvantageBatch(
            rewards=rewards, group_size=G, prompt_ids=torch.as_tensor(ids, dtype=torch.long),
            lengths=cmask.sum(dim=1).cpu(), step=step,
            seq_logps=(old_logps * cmask).sum(dim=1).cpu() if needs_logps else None,
        )
        advantages, keep = advantage(adv_batch)
    advantages = advantages.to(device)
    loss_mask = cmask if keep is None else cmask * keep.to(device).unsqueeze(1).to(cmask.dtype)
    if st.num_updates == 1 and needs_logps:
        old_logps = None     # computed only for the method; keep TRL's on-policy path for the loss

    # 5. Several updates on the same batch. Each micro-batch's loss is weighted so that the summed
    #    gradient equals the gradient of the full batch's loss.
    total_tokens = loss_mask.sum().clamp(min=1).item()
    for _ in range(st.num_updates):
        optimizer.zero_grad()
        loss_value, logps_parts, ent_parts = 0.0, [], []
        for s in pieces:
            logps, ent = per_token_logps_and_entropy(model, input_ids[s], attention_mask[s], C, st.temperature)
            loss = grpo_loss(logps, None if old_logps is None else old_logps[s], advantages[s], loss_mask[s],
                             ref_logps=None if ref_logps is None else ref_logps[s], beta=st.beta,
                             eps_low=st.eps_low, eps_high=st.eps_high, loss_type=st.loss_type,
                             max_completion_length=st.max_new_tokens)
            if len(pieces) > 1:
                if st.loss_type in ("bnpo", "dapo"):     # a mean over the batch's tokens
                    loss = loss * (loss_mask[s].sum().item() / total_tokens)
                else:                                     # a mean over the batch's completions
                    loss = loss * ((s.stop - s.start) / N)
            loss.backward()
            loss_value += loss.item()
            logps_parts.append(logps.detach())
            ent_parts.append(ent)
        if st.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), st.max_grad_norm).item()
        else:
            grad_norm = _grad_norm(model)
        optimizer.step()

    # 6. Statistics from the last update, over counted tokens only.
    with torch.no_grad():
        lp = torch.cat(logps_parts)
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
        entropy = ((torch.cat(ent_parts) * m).sum() / n_tokens).item()
        grouped_adv = advantages.detach().cpu().view(-1, G)
        zero_adv_frac = (grouped_adv.abs() < 1e-8).all(dim=1).float().mean().item()
        reward_var = rewards.view(-1, G).var(dim=1, unbiased=False).mean().item()
    stats = {"reward": rewards.mean().item(), "clip_frac": clip_frac, "kl": kl, "loss": loss_value,
             "completion_length": m.sum(dim=1).mean().item(), "zero_adv_frac": zero_adv_frac,
             "reward_var": reward_var, "entropy": entropy, "grad_norm": grad_norm, "rollouts": N}
    if "truncated" in batch:   # completions cut off before their answer could be scored
        stats["truncated_frac"] = batch["truncated"].float().mean().item()
    return stats


def _grad_norm(model) -> float:
    grads = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
    return torch.stack(grads).norm().item() if grads else 0.0
