"""GSM8K splits and the fixed few-shot prompt, identical for every run.

From the train split, with data.split_seed (never the run seed):
    shots: `prompt.shots` worked examples shown in every prompt
    val:   `data.val_size` problems for validation and tuning
    train: the rest. The run seed only shuffles their order.
The test split is used only for final evaluation.
"""
from __future__ import annotations

import json
import random
import re

# GSM8K solutions mark calculator calls like "<<48/2=24>>". Dropped from the few-shot answers.
_CALC = re.compile(r"<<[^>]*>>")


def load_gsm8k(dataset: str = "openai/gsm8k", local: dict | None = None) -> dict[str, list[dict]]:
    """{'train': [...], 'test': [...]}, each row {'question', 'answer'}.

    local: {'train': path.jsonl, 'test': path.jsonl} to read files instead of the Hub (tests, offline).
    """
    if local:
        out = {}
        for split, path in local.items():
            with open(path, encoding="utf-8") as f:
                out[split] = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset
        out = {split: list(load_dataset(dataset, "main", split=split)) for split in ("train", "test")}
    return {split: [{"question": r["question"], "answer": r["answer"]} for r in rows] for split, rows in out.items()}


def make_splits(train_rows: list[dict], val_size: int, shots: int, split_seed: int) -> dict[str, list[dict]]:
    """Fixed shots / val / train split of the train rows. Each row gets 'idx', its position in GSM8K's train split."""
    rows = [{**r, "idx": i} for i, r in enumerate(train_rows)]
    if shots + val_size >= len(rows):
        raise ValueError(f"shots + val_size = {shots + val_size} leaves no training problems out of {len(rows)}")
    order = list(range(len(rows)))
    random.Random(split_seed).shuffle(order)
    picked = [rows[i] for i in order]
    return {"shots": picked[:shots], "val": picked[shots:shots + val_size], "train": picked[shots + val_size:]}


def shuffled(rows: list[dict], seed: int) -> list[dict]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def clean_solution(answer: str) -> str:
    """A GSM8K reference solution as a few-shot answer: calculator marks removed, '#### N' kept."""
    return _CALC.sub("", answer).strip()


def build_messages(question: str, instruction: str, shots: list[dict]) -> list[dict]:
    """Chat turns: each worked example as a user turn and an assistant turn, then the question."""
    messages = []
    for ex in shots:
        messages.append({"role": "user", "content": f"{ex['question']} {instruction}"})
        messages.append({"role": "assistant", "content": clean_solution(ex["answer"])})
    messages.append({"role": "user", "content": f"{question} {instruction}"})
    return messages


def format_prompt(tokenizer, question: str, instruction: str, shots: list[dict]) -> str:
    return tokenizer.apply_chat_template(build_messages(question, instruction, shots),
                                         add_generation_prompt=True, tokenize=False)
