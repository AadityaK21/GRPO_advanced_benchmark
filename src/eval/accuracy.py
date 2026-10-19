"""Exact-match accuracy with greedy decoding: the primary metric's per-run number."""
from __future__ import annotations

import torch

from src.reward.gsm8k import exact_match_reward
from src.train.sample import _eos_ids, completion_mask


@torch.no_grad()
def greedy_completions(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int = 64) -> list[str]:
    """One greedy completion per prompt, decoded up to and including the first EOS."""
    was_training = model.training
    model.eval()          # generate() keeps its KV cache only in eval mode under gradient checkpointing
    try:
        device = next(model.parameters()).device
        eos = _eos_ids(model, tokenizer)
        # Similar lengths in each batch: less padding. The output goes back in the original order.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out: list[str | None] = [None] * len(prompts)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, padding_side="left")
            gen = model.generate(
                input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                max_new_tokens=max_new_tokens, do_sample=False,
                # Greedy, with Qwen's own sampling settings and repetition penalty switched off.
                temperature=None, top_k=None, top_p=None, repetition_penalty=1.0,
                eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
            )
            completions = gen[:, enc["input_ids"].size(1):]
            mask = completion_mask(completions, eos)
            for row, i in enumerate(idx):
                out[i] = tokenizer.decode(completions[row][: int(mask[row].sum())], skip_special_tokens=True)
        return out
    finally:
        model.train(was_training)


def evaluate(model, tokenizer, prompts: list[str], answers: list[str], max_new_tokens: int,
             batch_size: int = 64) -> dict:
    """{'accuracy': float, 'correct': [0/1 per problem], 'completions': [...]}."""
    completions = greedy_completions(model, tokenizer, prompts, max_new_tokens, batch_size)
    correct = [int(r) for r in exact_match_reward(completions, answers)]
    return {"accuracy": sum(correct) / max(len(correct), 1), "correct": correct, "completions": completions}
