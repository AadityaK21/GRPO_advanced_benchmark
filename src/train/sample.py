"""G completions per prompt, and the mask that says which completion tokens count."""
import torch


def completion_mask(completion_ids: torch.Tensor, eos_token_ids) -> torch.Tensor:
    """[B, C] tensor of 1s and 0s: 1 for each completion token up to and including the first EOS, 0 after.

    A row with no EOS (cut off at max_new_tokens) is all 1s.
    eos_token_ids: one id or a list of ids. Qwen2.5-Instruct stops on either of two,
    151645 (<|im_end|>) and 151643 (<|endoftext|>).
    """
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos = torch.tensor(list(eos_token_ids), device=completion_ids.device)
    is_eos = torch.isin(completion_ids, eos).long()          # 1 where the token is any EOS id
    # How many EOS tokens came strictly BEFORE each position. Zero means we have not yet
    # passed the first EOS, so the token counts; that includes the first EOS itself.
    eos_before = is_eos.cumsum(dim=1) - is_eos
    return (eos_before == 0).int()


@torch.no_grad()
def sample_group(model, tokenizer, prompts: list[str], group_size: int, max_new_tokens: int,
                 temperature: float) -> dict:
    """Sample `group_size` completions for each prompt.

    `prompts` are already formatted (chat template applied). Returns a dict of tensors on the
    model's device:
        "prompt_ids", "prompt_mask":         [N*G, P], prompts padded on the LEFT
        "completion_ids", "completion_mask": [N*G, C], with C <= max_new_tokens
    Rows i*G .. i*G+G-1 belong to prompts[i].

    Sample from softmax(logits / temperature) over the whole vocabulary: no top-k, no top-p and
    no repetition penalty, whatever the model's generation_config says. Your log-probs assume
    exactly this distribution. model.generate() is fine, but it fills in every setting you don't
    pass from the checkpoint's generation_config.json.
    """
    was_training = model.training
    model.eval()   # generate() drops its KV cache under gradient checkpointing in train mode: very slow
    try:
        return _sample(model, tokenizer, prompts, group_size, max_new_tokens, temperature)
    finally:
        model.train(was_training)


def _sample(model, tokenizer, prompts, group_size, max_new_tokens, temperature) -> dict:
    device = next(model.parameters()).device
    # Pad on the LEFT, so every prompt ends at the same column and every completion starts
    # right after it. Right padding would put pad tokens between a prompt and its answer.
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, padding_side="left")
    # Each prompt G times in a row: rows i*G .. i*G+G-1 are prompt i (repeat_interleave, not repeat,
    # which would tile the prompts and break the grouping that the advantages rely on).
    prompt_ids = enc["input_ids"].repeat_interleave(group_size, dim=0).to(device)
    prompt_mask = enc["attention_mask"].repeat_interleave(group_size, dim=0).to(device)
    eos = _eos_ids(model, tokenizer)
    out = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_new_tokens,
        # Plain sampling from softmax(logits / temperature). Every other rule is switched off
        # explicitly, because generate() would otherwise take Qwen's own top_k=20, top_p=0.8 and
        # repetition_penalty=1.1, and the log-probs in the loss would describe the wrong distribution.
        do_sample=True,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion_ids = out[:, prompt_ids.size(1):]          # generate() returns prompt + completion
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask(completion_ids, eos),
    }


def _eos_ids(model, tokenizer) -> list[int]:
    """Every id that ends a completion: the model's own list (Qwen2.5-Instruct has two) plus the tokenizer's."""
    ids = model.generation_config.eos_token_id
    ids = [] if ids is None else ([ids] if isinstance(ids, int) else list(ids))
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    return ids
