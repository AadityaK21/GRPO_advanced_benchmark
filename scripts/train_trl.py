"""The same run as scripts/train_grpo.py, but trained by TRL's GRPOTrainer (week 7).

It takes exactly the same flags. Same model, same prompts in the same order, same reward, same
settings, and every TRL option that our loop does not have is switched off. It writes
<out>/log.jsonl with the same fields, so scripts/compare_runs.py can put both on one plot.
The two runs will not match step for step, because the samples differ. Their seed bands should overlap.

--steps counts sampled batches, as in our loop; TRL counts optimizer steps, so it gets
steps * num_updates of them. With --num-updates 1 (the default) the two counts and the
learning-rate schedules are identical. With more, TRL's learning rate also decays between the
updates on one batch, while ours changes once per batch: a small, known difference.

    python scripts/train_trl.py --steps 100 --seed 0 --max-new-tokens 512 --lora-r 16 --lr 1e-5 --out runs/trl_seed0
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_grpo as ours  # noqa: E402  (the flags, the data order, the prompt and the dtype are shared)
from src.reward.gsm8k import exact_match_reward  # noqa: E402


def reward_fn(completions, answer, **kwargs) -> list[float]:
    """TRL hands over conversational completions ([{"role": "assistant", "content": ...}]) and the
    dataset's other columns by name. Score them with the same function as our loop."""
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return exact_match_reward(texts, answer)


class JsonlLog(TrainerCallback):
    """Write one line per optimizer step, with the same field names as scripts/train_grpo.py."""

    def __init__(self, path: Path, updates_per_batch: int = 1):
        self.f = open(path, "w", encoding="utf-8")
        self.k = updates_per_batch
        self.t = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "reward" not in logs:
            return
        now = time.perf_counter()
        row = {
            "step": (state.global_step - 1) // self.k,   # sampled-batch index, as in our loop
            "reward": logs["reward"],
            "completion_length": logs.get("completions/mean_length"),
            "truncated_frac": logs.get("completions/clipped_ratio"),   # TRL's name for "no EOS"
            "clip_frac": logs.get("clip_ratio/region_mean", 0.0),
            "kl": logs.get("kl", 0.0),
            "loss": logs.get("loss"),
            "lr": logs.get("learning_rate"),
            "seconds": round(now - self.t, 3),
            "trl": logs,   # everything TRL logged, unrenamed
        }
        if torch.cuda.is_available():
            row["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        self.t = now
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        print(f"step {row['step']:4d}  reward {row['reward']:.3f}  len {row['completion_length'] or 0:6.1f}  "
              f"cut {row['truncated_frac'] or 0:.2f}  "
              f"clip {row['clip_frac']:.3f}  kl {row['kl']:.4f}  {row['seconds']:.1f}s", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        self.f.close()


def main(argv=None):
    args = ours.parse_args(argv)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = ours.pick_dtype(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The same questions in the same order as our loop: our shuffle, then no shuffling inside TRL.
    examples = ours.load_examples(args)
    n = args.steps * args.prompts_per_step
    rows = [examples[i % len(examples)] for i in range(n)]
    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": f"{r['question']} {ours.INSTRUCTION}"}], "answer": r["answer"]}
        for r in rows
    ])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    peft_config = None
    if args.lora_r > 0:
        from peft import LoraConfig
        peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                                 target_modules="all-linear", task_type="CAUSAL_LM")

    config = GRPOConfig(
        output_dir=str(out / "trl"), report_to=[], save_strategy="no", logging_steps=1,
        max_steps=args.steps * args.num_updates, seed=args.seed, shuffle_dataset=False,
        per_device_train_batch_size=args.prompts_per_step * args.group_size, num_generations=args.group_size,
        gradient_accumulation_steps=1, max_completion_length=args.max_new_tokens,
        learning_rate=args.lr, lr_scheduler_type="linear", warmup_steps=0, weight_decay=0.0,
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, max_grad_norm=args.max_grad_norm,
        num_iterations=args.num_updates, loss_type=args.loss_type, beta=args.beta,
        epsilon=args.eps_low, epsilon_high=args.eps_high, scale_rewards=args.scale,
        temperature=args.temperature, top_k=0, top_p=1.0, repetition_penalty=1.0,
        # TRL options our loop does not have, switched off:
        use_bias_correction_kl=False, mask_truncated_completions=False, importance_sampling_level="token",
        top_entropy_quantile=1.0, entropy_coef=0.0, disable_dropout=True,
        gradient_checkpointing=not args.no_grad_ckpt, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16, fp16=False, use_cpu=device == "cpu",
    )
    (out / "config.json").write_text(json.dumps({**vars(args), "trainer": "trl", "instruction": ours.INSTRUCTION,
                                                 "device": device, "dtype": str(dtype)}, indent=2))
    trainer = GRPOTrainer(model=model, args=config, processing_class=tokenizer, reward_funcs=reward_fn,
                          train_dataset=dataset, peft_config=peft_config, callbacks=[JsonlLog(out / "log.jsonl", args.num_updates)])
    from transformers.trainer_callback import PrinterCallback, ProgressCallback
    trainer.remove_callback(PrinterCallback)    # our callback prints one line per step instead
    trainer.remove_callback(ProgressCallback)
    trainer.train()
    if args.save:
        trainer.save_model(str(out / "model"))


if __name__ == "__main__":
    main()
