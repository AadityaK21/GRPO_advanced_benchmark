"""Loading the policy: dtype, gradient checkpointing, LoRA, and the reference model."""
from __future__ import annotations

import copy

import torch


def pick_dtype(device: str, requested: str = "auto") -> torch.dtype:
    """bf16 only where the GPU has it natively (A100, L4, RTX 30/40...). Kaggle's T4 does not: fp32 there."""
    if requested != "auto":
        return {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[requested]
    if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float32


def load_policy(name: str, device: str, *, dtype: str = "auto", tokenizer_name: str | None = None,
                gradient_checkpointing: bool = True, lora_r: int = 0, lora_alpha: int | None = None,
                lora_dropout: float = 0.0, lora_target_modules="all-linear", need_ref: bool = False):
    """(model in train mode, tokenizer, ref_model or None). lora_r = 0 means full fine-tuning."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=pick_dtype(device, dtype)).to(device)
    if gradient_checkpointing:
        # Keep only each layer's input during the forward pass and recompute the rest in backward.
        # It only acts in train mode; sampling switches to eval mode by itself.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ref_model = None
    if lora_r > 0:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=lora_dropout,
                          target_modules=lora_target_modules, task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        if need_ref:
            ref_model = AdapterOff(model)
    elif need_ref:
        ref_model = copy.deepcopy(model).eval().requires_grad_(False)
    model.train()   # Qwen2.5 has no dropout, so train and eval mode give the same log-probs
    return model, tokenizer, ref_model


class AdapterOff(torch.nn.Module):
    """The reference model for LoRA runs: the same model with the adapter switched off, so no second
    copy sits in memory. per_token_logps calls it with gradients off. get_decoder and
    get_output_embeddings let it score in chunks like the model itself."""

    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model

    def forward(self, *args, **kwargs):
        with self.peft_model.disable_adapter():
            return self.peft_model(*args, **kwargs)

    def get_decoder(self):
        return _WithAdapterOff(self.peft_model, self.peft_model.get_decoder())

    def get_output_embeddings(self):
        return _WithAdapterOff(self.peft_model, self.peft_model.get_output_embeddings())


class _WithAdapterOff(torch.nn.Module):
    """Runs one part of a PEFT model (its body or its output layer) with the adapter switched off."""

    def __init__(self, peft_model, part):
        super().__init__()
        self._peft = [peft_model]     # a list, so it is not registered as a submodule twice
        self.part = part

    def forward(self, *args, **kwargs):
        with self._peft[0].disable_adapter():
            return self.part(*args, **kwargs)
