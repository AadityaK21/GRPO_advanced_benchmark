"""pass@k on the test split: the secondary metric.

For each test problem, sample n completions at the training temperature (plain softmax sampling, the
same sampler as training) and count how many are right. pass@k is then estimated per problem with the
unbiased estimator of Chen et al. 2021 (arXiv 2107.03374, eq. 1),

    pass@k = 1 - C(n - c, k) / C(n, k)        n samples, c of them right,

and averaged over problems. With n = 16 this gives pass@1, pass@4 and pass@16 from the same samples.
pass@1 here is the mean over sampled completions, not greedy accuracy (the primary metric).

Scoring is the same as the greedy test evaluation (src/eval/accuracy.py): exact match on the decoded
text, no extra rule for completions cut off at max_new_tokens. So the two test metrics differ only in
how completions are decoded. (Training's `reward.truncated_is_wrong` is a training choice.)

Why it is reported: RLVR can raise pass@1 while lowering pass@k for large k (2607.20543), so a method
that wins on greedy accuracy is checked for a hidden loss in coverage.
"""
from __future__ import annotations

import time
from typing import Callable

import torch

from src.reward.gsm8k import exact_match_reward
from src.train.sample import sample_group


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: n samples, c correct. 1 - C(n-c, k) / C(n, k), as a stable product."""
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:                      # every k-subset contains a correct sample
        return 1.0
    prob_all_wrong = 1.0
    for i in range(n - c + 1, n + 1):  # C(n-c, k) / C(n, k) = prod_{i=n-c+1}^{n} (1 - k / i)
        prob_all_wrong *= 1.0 - k / i
    return 1.0 - prob_all_wrong


def ks_for(n: int) -> list[int]:
    """The k values reported for n samples: 1 and every power of 4 up to n, plus n itself (16 -> 1, 4, 16)."""
    ks, k = [1], 4
    while k <= n:
        ks.append(k)
        k *= 4
    if ks[-1] != n:
        ks.append(n)
    return ks


@torch.no_grad()
def sampled_correct_counts(model, tokenizer, prompts: list[str], answers: list[str], n: int,
                           max_new_tokens: int, temperature: float, truncated_is_wrong: bool = False,
                           prompts_per_batch: int = 8, log: Callable[[str], None] | None = None) -> list[int]:
    """Number of correct completions out of n, per prompt. Same sampler and reward as training."""
    counts: list[int] = []
    n_batches = (len(prompts) + prompts_per_batch - 1) // prompts_per_batch
    t_start = time.perf_counter()
    for start in range(0, len(prompts), prompts_per_batch):
        chunk = prompts[start:start + prompts_per_batch]
        batch = sample_group(model, tokenizer, chunk, n, max_new_tokens, temperature)
        ids, mask = batch["completion_ids"], batch["completion_mask"]
        texts = [tokenizer.decode(ids[r][: int(mask[r].sum())], skip_special_tokens=True) for r in range(ids.size(0))]
        per_row_answers = [a for a in answers[start:start + len(chunk)] for _ in range(n)]
        rewards = exact_match_reward(texts, per_row_answers)
        if truncated_is_wrong:
            rewards = [0.0 if cut else r for r, cut in zip(rewards, batch["truncated"].tolist())]
        for i in range(len(chunk)):
            counts.append(int(sum(r > 0 for r in rewards[i * n:(i + 1) * n])))
        if log is not None:
            log(f"  pass@k batch {start // prompts_per_batch + 1}/{n_batches}: {len(counts)}/{len(prompts)} problems, "
                f"{time.perf_counter() - t_start:.0f}s so far")
    return counts


def summarise(counts: list[int], n: int) -> dict:
    """{'pass@1': ..., 'pass@4': ..., 'pass@16': ...} averaged over problems."""
    if not counts:
        return {}
    return {f"pass@{k}": sum(pass_at_k(n, c, k) for c in counts) / len(counts) for k in ks_for(n)}


def evaluate_passk(model, tokenizer, prompts: list[str], answers: list[str], n: int, max_new_tokens: int,
                   temperature: float, truncated_is_wrong: bool = False, prompts_per_batch: int = 8,
                   seed: int | None = None, log: Callable[[str], None] | None = None) -> dict:
    """pass@k for every k in ks_for(n), plus the per-problem counts (for paired tests and figures).

    seed: re-seeds torch first, so the samples do not depend on how much random state training used.
    """
    if seed is not None:
        torch.manual_seed(seed)
    counts = sampled_correct_counts(model, tokenizer, prompts, answers, n, max_new_tokens, temperature,
                                    truncated_is_wrong, prompts_per_batch, log)
    return {"samples": n, "temperature": temperature, "problems": len(counts), **summarise(counts, n),
            "correct_counts": counts}
