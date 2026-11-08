"""pass@k: the estimator against brute force, and the sampler on the tiny model."""
import itertools
import math

import pytest

from src.eval.passk import evaluate_passk, ks_for, pass_at_k, summarise
from tiny import make_model, make_tokenizer


def brute_force(n, c, k):
    """Share of the k-subsets of n samples (c right) that contain at least one right sample."""
    samples = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(any(samples[i] for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_pass_at_k_equals_brute_force(n):
    for c in range(n + 1):
        for k in range(1, n + 1):
            assert pass_at_k(n, c, k) == pytest.approx(brute_force(n, c, k), abs=1e-12), (n, c, k)


def test_pass_at_k_edge_cases_and_large_n():
    assert pass_at_k(16, 0, 16) == 0.0 and pass_at_k(16, 1, 16) == 1.0
    assert pass_at_k(16, 4, 1) == pytest.approx(4 / 16), "pass@1 is the share of right samples"
    # The product form is exact: compare with the binomial formula at n = 16.
    for c in range(17):
        for k in (1, 4, 16):
            exact = 1 - (math.comb(16 - c, k) / math.comb(16, k) if 16 - c >= k else 0)
            assert pass_at_k(16, c, k) == pytest.approx(exact, abs=1e-12)
    with pytest.raises(ValueError):
        pass_at_k(4, 5, 1)
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 5)


def test_ks_and_summary():
    assert ks_for(16) == [1, 4, 16] and ks_for(4) == [1, 4] and ks_for(8) == [1, 4, 8] and ks_for(1) == [1]
    s = summarise([0, 4, 16, 1], n=16)
    assert set(s) == {"pass@1", "pass@4", "pass@16"}
    assert s["pass@1"] == pytest.approx((0 + 4 + 16 + 1) / 64)
    assert s["pass@16"] == pytest.approx(3 / 4), "pass@n: share of problems with any right sample"
    assert s["pass@1"] <= s["pass@4"] <= s["pass@16"]
    assert summarise([], 16) == {}


def test_evaluate_passk_on_the_tiny_model(monkeypatch):
    """Counts are per prompt, bounded by n, repeatable under a seed, and scored by the given answers."""
    model, tok = make_model(seed=0, logit_scale=4.0), make_tokenizer()
    prompts = [f"t{k} t{k + 1}" for k in range(5)]
    # The reward matches the completion's last number against the answer; the tiny model writes tokens
    # "tK", which have no digits, so patch the scorer to: right iff the first token is t3.
    import src.eval.passk as passk_mod
    monkeypatch.setattr(passk_mod, "exact_match_reward",
                        lambda texts, answers: [1.0 if t.split()[:1] == ["t3"] else 0.0 for t in texts])
    a = evaluate_passk(model, tok, prompts, ["#### 0"] * 5, n=8, max_new_tokens=2, temperature=1.0,
                       prompts_per_batch=2, seed=7)
    b = evaluate_passk(model, tok, prompts, ["#### 0"] * 5, n=8, max_new_tokens=2, temperature=1.0,
                       prompts_per_batch=3, seed=7)
    assert len(a["correct_counts"]) == 5 and all(0 <= c <= 8 for c in a["correct_counts"])
    assert a["samples"] == 8 and a["problems"] == 5 and set(k for k in a if k.startswith("pass@")) == {"pass@1", "pass@4", "pass@8"}
    assert a["pass@1"] <= a["pass@4"] <= a["pass@8"]
    assert a == evaluate_passk(model, tok, prompts, ["#### 0"] * 5, n=8, max_new_tokens=2, temperature=1.0,
                               prompts_per_batch=2, seed=7), "same seed, same samples"
    assert b["problems"] == 5, "batching changes the random stream, not the bookkeeping"
    # The counting itself: a scorer that is always right gives n per prompt, never right gives 0.
    lines = []
    monkeypatch.setattr(passk_mod, "exact_match_reward", lambda texts, answers: [1.0] * len(texts))
    sure = evaluate_passk(model, tok, prompts[:2], ["#### 0"] * 2, n=4, max_new_tokens=1, temperature=1.0,
                          prompts_per_batch=1, seed=0, log=lines.append)
    assert sure["correct_counts"] == [4, 4] and sure["pass@1"] == 1.0
    assert len(lines) == 2 and "2/2 problems" in lines[-1], "one progress line per batch"
    monkeypatch.setattr(passk_mod, "exact_match_reward", lambda texts, answers: [0.0] * len(texts))
    none = evaluate_passk(model, tok, prompts[:2], ["#### 0"] * 2, n=4, max_new_tokens=1, temperature=1.0, seed=0)
    assert none["correct_counts"] == [0, 0] and none["pass@4"] == 0.0


def test_answers_line_up_with_their_prompts(monkeypatch):
    """Each prompt's n samples are scored against that prompt's answer, across batch boundaries."""
    import src.eval.passk as passk_mod
    model, tok = make_model(seed=1), make_tokenizer()
    monkeypatch.setattr(passk_mod, "exact_match_reward",
                        lambda texts, answers: [1.0 if a == "#### 1" else 0.0 for a in answers])
    res = evaluate_passk(model, tok, [f"t{k}" for k in range(5)], ["#### 1", "#### 0", "#### 1", "#### 1", "#### 0"],
                         n=3, max_new_tokens=1, temperature=1.0, prompts_per_batch=2, seed=0)
    assert res["correct_counts"] == [3, 0, 3, 3, 0]
