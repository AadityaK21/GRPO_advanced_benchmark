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
    raise NotImplementedError
