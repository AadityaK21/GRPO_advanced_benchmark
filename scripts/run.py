"""Run one benchmark condition: one method, one group size, one seed (the weeks 8-9 harness).

    python scripts/run.py --method vanilla --seed 0
    python scripts/run.py --method dr_grpo --seed 3 --set budget.group_size=4
    python scripts/run.py --eval-only --seed 0          # the untrained base model, evaluated identically

Settings come from configs/base.yaml, then configs/methods/<method>.yaml, then each --set key=value.
The run writes results/<method>/g<G>/seed<seed>-<time>/ (see src/runlog.py): config, seed, git
commit, a line per step, validation checks, and summary.json with the test accuracy at the end.

Smoke test of the whole pipeline (2 short steps, 16 validation and 16 test problems, into runs/):
    python scripts/run.py --method vanilla --seed 0 --smoke

What a run does:
  1. Splits GSM8K train into fixed few-shot examples, a fixed validation set and the training pool
     (data.split_seed: identical for every run). The run seed shuffles the training pool and
     seeds LoRA's initialisation and the sampling.
  2. Trains for budget.total_rollouts completions, budget.rollouts_per_step at a time.
  3. Checks greedy validation accuracy before training and every eval.every steps, keeping the best
     step's weights.
  4. Evaluates the final and the best weights on the test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before CUDA starts

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.advantages.base import build_advantage  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import format_prompt, load_gsm8k, make_splits, shuffled  # noqa: E402
from src.eval.accuracy import evaluate  # noqa: E402
from src.reward.gsm8k import exact_match_reward  # noqa: E402
from src.runlog import JsonlWriter, new_run_dir, write_json, write_start  # noqa: E402
from src.train.loop import GRPOSettings, grpo_step  # noqa: E402
from src.train.model import load_policy, pick_dtype  # noqa: E402

SMOKE = ["budget.rollouts_per_step=16", "budget.total_rollouts=32", "budget.micro_batch_size=8",
         "sampling.max_new_tokens=128", "eval.every=1", "eval.val_problems=16", "eval.test_problems=16",
         "eval.max_new_tokens=128", "results_dir=runs/smoke"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="vanilla", help="a file name in configs/methods/, without .yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set budget.group_size=4 (repeatable)")
    p.add_argument("--eval-only", action="store_true", help="no training: evaluate the base model (results/base/)")
    p.add_argument("--smoke", action="store_true", help="tiny budget and eval sets, written under runs/smoke")
    p.add_argument("--no-ckpt", action="store_true", help="do not save the best and final weights")
    p.add_argument("--local-data", nargs=2, metavar=("TRAIN_JSONL", "TEST_JSONL"),
                   help="read GSM8K from local files instead of the Hub (tests, offline machines)")
    return p.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    cfg = load_config(args.method, list(args.set) + (SMOKE if args.smoke else []))
    if args.eval_only:
        cfg["method"] = "base"      # the untrained model: same prompt, same evaluation, its own folder
    # Build the method first, so a bad method or group size fails before any download or model load.
    advantage = build_advantage(cfg["advantage"]["name"], cfg["advantage"].get("params"))
    advantage.check_group_size(cfg["budget"]["group_size"])
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)        # LoRA initialisation and the sampling stream
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data -----------------------------------------------------------------------------------
    local = {"train": args.local_data[0], "test": args.local_data[1]} if args.local_data else None
    data = load_gsm8k(cfg["data"]["dataset"], local=local)
    splits = make_splits(data["train"], cfg["data"]["val_size"], cfg["prompt"]["shots"], cfg["data"]["split_seed"])
    train_pool = shuffled(splits["train"], seed)
    val = splits["val"][: cfg["eval"]["val_problems"]]
    test = data["test"][: cfg["eval"]["test_problems"]] if cfg["eval"]["test_problems"] else data["test"]

    # ---- model ----------------------------------------------------------------------------------
    m, lora = cfg["model"], cfg["lora"]
    model, tokenizer, ref_model = load_policy(
        m["name"], device, dtype=m["dtype"], tokenizer_name=m.get("tokenizer"),
        gradient_checkpointing=m["gradient_checkpointing"], lora_r=lora["r"], lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"], lora_target_modules=lora["target_modules"],
        need_ref=cfg["loss"]["beta"] != 0.0 and not args.eval_only)

    instruction, shots = cfg["prompt"]["instruction"], splits["shots"]

    def prompts_for(rows):
        return [format_prompt(tokenizer, r["question"], instruction, shots) for r in rows]

    val_prompts, val_answers = prompts_for(val), [r["answer"] for r in val]
    test_prompts, test_answers = prompts_for(test), [r["answer"] for r in test]

    # ---- budget ---------------------------------------------------------------------------------
    b = cfg["budget"]
    cluster_of = None
    if "cluster_ids" in advantage.needs and not args.eval_only:
        path = Path(cfg["data"]["clusters_file"])
        cluster_of = load_clusters(path if path.is_absolute() else ROOT / path, [r["question"] for r in data["train"]])
    per_prompt = b["group_size"] + advantage.extra_rollouts
    prompts_per_step = max(1, b["rollouts_per_step"] // per_prompt)
    rollouts_per_step = prompts_per_step * per_prompt
    dynamic = bool(cfg["filter"]["drop_uniform_groups"])
    # Without dynamic sampling every step costs exactly rollouts_per_step. With it (DAPO) a step costs
    # that plus every refill, so the run goes until the rollout budget is spent: `steps` is then only
    # the most it could take.
    steps = 0 if args.eval_only else b["total_rollouts"] // rollouts_per_step

    run_dir = new_run_dir(cfg, seed)
    write_start(run_dir, cfg, seed, extra={
        "device": device, "dtype": str(pick_dtype(device, m["dtype"])), "steps": steps,
        "prompts_per_step": prompts_per_step, "rollouts_per_step": rollouts_per_step,
        "splits": {"shots": [r["idx"] for r in shots], "val": len(splits["val"]), "train": len(train_pool),
                   "test": len(data["test"])},
        "val_ids_sha1": hashlib.sha1(",".join(str(r["idx"]) for r in splits["val"]).encode()).hexdigest(),
        "example_prompt": val_prompts[0] if val_prompts else None,
    })
    print(f"run dir {run_dir}\n{'at most ' if dynamic else ''}{steps} steps x {prompts_per_step} prompts x "
          f"{per_prompt} completions = {steps * rollouts_per_step} rollouts on {device}"
          + (" (dynamic sampling: refills count against the budget)" if dynamic else ""), flush=True)

    ec = cfg["eval"]
    ckpt_dir = run_dir / "ckpt"
    t_start = time.perf_counter()
    eval_seconds = 0.0
    eval_log = JsonlWriter(run_dir / "eval.jsonl")
    best = {"step": None, "val_accuracy": -1.0}
    best_weights = None

    def check_val(step):
        nonlocal eval_seconds, best_weights
        t0 = time.perf_counter()
        res = evaluate(model, tokenizer, val_prompts, val_answers, ec["max_new_tokens"], ec["batch_size"])
        eval_seconds += time.perf_counter() - t0
        eval_log.write({"step": step, "val_accuracy": res["accuracy"], "val_format_rate": res["format_rate"],
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"step {step:4d}  val accuracy {res['accuracy']:.4f}  format {res['format_rate']:.2f}", flush=True)
        if res["accuracy"] > best["val_accuracy"]:          # ties keep the earlier step
            best.update(step=step, val_accuracy=res["accuracy"])
            best_weights = _trainable_state(model)
        return res["accuracy"]

    base_val = check_val(0)
    last_val = base_val

    # ---- training -------------------------------------------------------------------------------
    rollouts = 0
    if steps:
        o = cfg["optimizer"]
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                                      weight_decay=o["weight_decay"])
        if o["schedule"] != "linear":
            raise ValueError(f"optimizer.schedule {o['schedule']!r} is not implemented")
        budget = b["total_rollouts"]
        # Linear decay to 0 over the run: by step for a fixed cost per step, by rollouts spent otherwise.
        progress = (lambda s: rollouts / budget) if dynamic else (lambda s: s / steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1.0 - progress(s)))
        L = cfg["loss"]
        settings = GRPOSettings(
            group_size=b["group_size"], max_new_tokens=cfg["sampling"]["max_new_tokens"],
            temperature=cfg["sampling"]["temperature"], num_updates=L["num_updates"], loss_type=L["loss_type"],
            eps_low=L["eps_low"], eps_high=L["eps_high"], beta=L["beta"], max_grad_norm=o["max_grad_norm"],
            micro_batch_size=b["micro_batch_size"], truncated_is_wrong=cfg["reward"]["truncated_is_wrong"],
            drop_uniform_groups=dynamic, max_refills=cfg["filter"]["max_refills"],
        )
        train_log = JsonlWriter(run_dir / "train.jsonl")
        cursor, step_times = 0, []

        def next_rows(n):
            nonlocal cursor
            rows = [train_pool[(cursor + i) % len(train_pool)] for i in range(n)]
            cursor += n
            return rows

        def refill(n):
            rows = next_rows(n)
            return prompts_for(rows), [r["answer"] for r in rows], [r["idx"] for r in rows]

        step, last_checked = 0, 0
        while step < steps and rollouts + rollouts_per_step <= b["total_rollouts"]:
            rows = next_rows(prompts_per_step)
            lr = scheduler.get_last_lr()[0]
            t0 = time.perf_counter()
            stats = grpo_step(model, optimizer, tokenizer, prompts_for(rows), [r["answer"] for r in rows],
                              exact_match_reward, settings, ref_model, advantage=advantage,
                              prompt_ids=[r["idx"] for r in rows], step=step,
                              refill=refill if dynamic else None,
                              max_rollouts=b["total_rollouts"] - rollouts, cluster_of=cluster_of)
            rollouts += stats["rollouts"]
            scheduler.step()
            step_times.append(time.perf_counter() - t0)
            stats.update(step=step + 1, rollouts_total=rollouts, lr=lr, seconds=round(step_times[-1], 2))
            if device == "cuda":
                stats["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            train_log.write(stats)
            step += 1
            if stats.get("skipped"):
                print(f"step {step:4d}  every group had equal rewards after {stats['refills']} refills: "
                      f"no update  ({rollouts} rollouts spent)", flush=True)
                continue
            left = (b["total_rollouts"] - rollouts) / max(rollouts / step, 1) if dynamic else steps - step
            eta = sum(step_times) / len(step_times) * left / 60
            print(f"step {step:4d}/{steps}  reward {stats['reward']:.3f}  uniform {stats['uniform_frac']:.2f}  "
                  f"zero-adv {stats['zero_adv_frac']:.2f}  "
                  f"len {stats['completion_length']:5.0f}  ent {stats['entropy']:.3f}  "
                  f"gnorm {stats['grad_norm']:.3f}  {stats['seconds']:.1f}s  ~{eta:.0f} min left"
                  + (f"  kept {stats['groups_kept']}, dropped {stats['groups_dropped']}" if dynamic else "")
                  + (f"  peak {stats['gpu_peak_gb']:.1f} GB" if "gpu_peak_gb" in stats else ""), flush=True)
            if not all(map(_finite, (stats["loss"], stats["grad_norm"]))):
                raise FloatingPointError(f"non-finite loss or gradient at step {step}; see {run_dir}")
            if step % ec["every"] == 0:
                last_val, last_checked = check_val(step), step
        if last_checked != step:              # always validate the final weights
            last_val = check_val(step)
        steps = step                          # the steps actually taken
        train_log.close()
    train_seconds = time.perf_counter() - t_start - eval_seconds

    # ---- test -----------------------------------------------------------------------------------
    t0 = time.perf_counter()
    final = evaluate(model, tokenizer, test_prompts, test_answers, ec["max_new_tokens"], ec["batch_size"])
    if not args.no_ckpt and steps:
        _save(model, ckpt_dir / "final")
    best_test = final
    if best["step"] != steps:                 # the best validation step is not the last one
        _load_trainable_state(model, best_weights)
        best_test = evaluate(model, tokenizer, test_prompts, test_answers, ec["max_new_tokens"], ec["batch_size"])
        if not args.no_ckpt:
            _save(model, ckpt_dir / "best")
    eval_seconds += time.perf_counter() - t0
    (ckpt_dir).mkdir(exist_ok=True)
    (ckpt_dir / "test_completions_final.txt").write_text("\n\n=====\n\n".join(final["completions"]), encoding="utf-8")

    summary = {
        "method": cfg["method"], "seed": seed, "group_size": b["group_size"], "steps": steps,
        "rollouts": rollouts, "budget_rollouts": 0 if args.eval_only else b["total_rollouts"],
        "base_val_accuracy": base_val,
        "final": {"step": steps, "val_accuracy": last_val, "test_accuracy": final["accuracy"],
                  "test_format_rate": final["format_rate"]},
        "best": {"step": best["step"], "val_accuracy": best["val_accuracy"], "test_accuracy": best_test["accuracy"],
                 "test_format_rate": best_test["format_rate"]},
        "test_problems": len(test),
        "test_correct_final": "".join(map(str, final["correct"])),
        "test_correct_best": "".join(map(str, best_test["correct"])),
        "train_seconds": round(train_seconds, 1), "eval_seconds": round(eval_seconds, 1),
        "wall_clock_seconds": round(time.perf_counter() - t_start, 1),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None,
    }
    eval_log.close()
    write_json(run_dir / "summary.json", summary)
    print(f"test accuracy: final {final['accuracy']:.4f}, best (step {best['step']}) {best_test['accuracy']:.4f}  "
          f"| {summary['wall_clock_seconds'] / 60:.1f} min  | {run_dir}", flush=True)
    return run_dir


def load_clusters(path: Path, train_questions: list[str]) -> torch.Tensor:
    """cluster_of[i] = the cluster of GSM8K train problem i, from the file scripts/make_clusters.py writes.
    Refuses a file made from different questions (it records a hash of them)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python scripts/make_clusters.py` once (needs the "
                                "sentence-transformers and scikit-learn packages)")
    info = json.loads(path.read_text())
    ids = torch.tensor(info["cluster_ids"], dtype=torch.long)
    sha = hashlib.sha1("\n".join(train_questions).encode("utf-8")).hexdigest()
    if ids.numel() != len(train_questions) or info.get("questions_sha1") != sha:
        raise ValueError(f"{path} was made from other questions ({ids.numel()} ids) than this run's GSM8K train "
                         f"({len(train_questions)} problems): re-run scripts/make_clusters.py")
    return ids


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def _trainable_state(model) -> dict:
    """A CPU copy of the weights being trained (the adapter, for LoRA)."""
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}


def _load_trainable_state(model, state: dict) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n].to(p.device))


def _save(model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "peft_config"):
        model.save_pretrained(path)                     # the adapter only, tens of MB
    else:
        torch.save(_trainable_state(model), path / "weights.pt")


if __name__ == "__main__":
    main()
