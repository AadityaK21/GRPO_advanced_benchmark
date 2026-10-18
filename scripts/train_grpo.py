"""Train Qwen2.5-0.5B-Instruct on GSM8K with your own GRPO loop (weeks 6-7).

Every default below is TRL 1.15.0's GRPOTrainer default (see the table in docs/grpo-loop.md),
so this script and TRL started with the same flags are comparable. Each step writes one JSON
line to <out>/log.jsonl: reward, clip fraction, KL, loss, completion length, learning rate, time.

On Kaggle, from the repo folder:
    !pip install -q transformers==5.19.0 trl==1.15.0 datasets peft
    !python scripts/train_grpo.py --steps 100 --seed 0 --out runs/ours_seed0

Short smoke run anywhere (CPU is fine, just slow):
    python scripts/train_grpo.py --steps 2 --max-new-tokens 32 --out runs/smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation; before CUDA starts

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import src...` works from anywhere

from src.reward.gsm8k import exact_match_reward  # noqa: E402
from src.train.loop import GRPOSettings, grpo_step  # noqa: E402

# Appended to every question. A choice that can change results: record it in implementation-notes.md.
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--data", default=None,
                   help="a local .jsonl with 'question' and 'answer' fields instead of GSM8K from the Hub")
    p.add_argument("--steps", type=int, default=100, help="sampled batches, each followed by --num-updates updates")
    p.add_argument("--prompts-per-step", type=int, default=1,
                   help="TRL: per_device_train_batch_size / num_generations = 8 / 8 = 1")
    p.add_argument("--group-size", type=int, default=8, help="TRL: num_generations")
    p.add_argument("--max-new-tokens", type=int, default=512, help="TRL: max_completion_length")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-6, help="AdamW, decayed linearly to 0 over --steps")
    p.add_argument("--num-updates", type=int, default=1, help="TRL: num_iterations")
    p.add_argument("--loss-type", default="dapo", choices=["grpo", "bnpo", "dapo", "dr_grpo"])
    p.add_argument("--scale", default="group", choices=["group", "none"], help="TRL: scale_rewards")
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    p.add_argument("--eps-low", type=float, default=0.2, help="TRL: epsilon")
    p.add_argument("--eps-high", type=float, default=0.2, help="TRL: epsilon_high (None means epsilon)")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=0, help="0 = full fine-tuning; >0 = LoRA of this rank")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="turn off gradient checkpointing (TRL's default is on; it trades compute for memory)")
    p.add_argument("--seed", type=int, default=42, help="TRL's default seed")
    p.add_argument("--no-shuffle", action="store_true", help="keep the dataset order (TRL: shuffle_dataset=False)")
    p.add_argument("--out", default="runs/grpo")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save", action="store_true", help="save the model (or adapter) to <out>/model at the end")
    return p.parse_args(argv)


def load_examples(args) -> list[dict]:
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset
        rows = list(load_dataset("openai/gsm8k", "main", split="train"))
    rows = [{"question": r["question"], "answer": r["answer"]} for r in rows]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)
    return rows


def format_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": f"{question} {INSTRUCTION}"}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


class AdapterOff(torch.nn.Module):
    """The reference model for LoRA runs: the same model with the adapter switched off, so no second
    copy sits in memory. per_token_logps only calls it, and a no-grad call is all it needs."""

    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model

    def forward(self, *args, **kwargs):
        with self.peft_model.disable_adapter():
            return self.peft_model(*args, **kwargs)


def pick_dtype(device: str) -> torch.dtype:
    # bf16 only where the GPU has it natively (A100, L4, RTX 30/40...). Kaggle's T4 does not: use fp32 there.
    if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float32


def build(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=pick_dtype(device)).to(device)
    if not args.no_grad_ckpt:
        # Keep only each layer's input during the forward pass and recompute the rest in backward.
        # It only acts in train mode; sampling switches to eval mode by itself.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ref_model = None
    if args.lora_r > 0:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,  # dropout 0: log-probs
                          target_modules="all-linear", task_type="CAUSAL_LM")             # must be deterministic
        model = get_peft_model(model, lora)
        if args.beta != 0.0:
            ref_model = AdapterOff(model)
    elif args.beta != 0.0:
        ref_model = copy.deepcopy(model).eval().requires_grad_(False)
    model.train()   # Qwen2.5 has no dropout, so train and eval mode give the same log-probs
    return model, tokenizer, ref_model


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, ref_model = build(args, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    # TRL's "linear" schedule with 0 warmup: lr * (1 - step / total). One scheduler step per sampled
    # batch, which matches TRL's count only while --num-updates is 1.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1.0 - s / args.steps))

    settings = GRPOSettings(
        group_size=args.group_size, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        num_updates=args.num_updates, loss_type=args.loss_type, eps_low=args.eps_low, eps_high=args.eps_high,
        beta=args.beta, scale=args.scale, max_grad_norm=args.max_grad_norm,
    )
    examples = load_examples(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "instruction": INSTRUCTION, "device": device,
                                                 "dtype": str(pick_dtype(device))}, indent=2))

    history, cursor = [], 0
    with open(out / "log.jsonl", "w", encoding="utf-8") as log:
        for step in range(args.steps):
            batch = [examples[(cursor + i) % len(examples)] for i in range(args.prompts_per_step)]
            cursor += args.prompts_per_step
            prompts = [format_prompt(tokenizer, ex["question"]) for ex in batch]
            answers = [ex["answer"] for ex in batch]

            lr = scheduler.get_last_lr()[0]
            t0 = time.perf_counter()
            stats = grpo_step(model, optimizer, tokenizer, prompts, answers, exact_match_reward, settings, ref_model)
            scheduler.step()
            stats.update(step=step, lr=lr, seconds=round(time.perf_counter() - t0, 3))
            if device == "cuda":
                stats["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            history.append(stats)
            log.write(json.dumps(stats) + "\n")
            log.flush()
            if step % args.log_every == 0 or step == args.steps - 1:
                print(f"step {step:4d}  reward {stats['reward']:.3f}  len {stats['completion_length']:6.1f}  "
                      f"cut {stats.get('truncated_frac', 0.0):.2f}  "
                      f"clip {stats['clip_frac']:.3f}  kl {stats['kl']:.4f}  {stats['seconds']:.1f}s"
                      + (f"  peak {stats['gpu_peak_gb']:.1f} GB" if "gpu_peak_gb" in stats else ""), flush=True)

    if args.save:
        model.save_pretrained(out / "model")
        tokenizer.save_pretrained(out / "model")
    return history


if __name__ == "__main__":
    main()
