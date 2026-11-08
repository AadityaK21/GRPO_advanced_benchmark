"""Checks for scripts/aggregate.py on fake finished runs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("aggregate", ROOT / "scripts" / "aggregate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(results, method, g, seed, acc, stamp="20261101-000000", dirty=False, passk=None):
    d = results / method / f"g{g}" / f"seed{seed}-{stamp}"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"seed": seed, "gpu": "A100", "dtype": "torch.bfloat16",
                                             "commit": "abc", "dirty": dirty}))
    (d / "summary.json").write_text(json.dumps({
        "method": method, "seed": seed, "group_size": g, "rollouts": 12800, "wall_clock_seconds": 600,
        "final": {"test_accuracy": acc}, "best": {"test_accuracy": acc + 0.01}, "passk": passk}))
    return d


def test_aggregate_ranks_compares_and_flags(tmp_path):
    rng = np.random.default_rng(0)
    seed_effect = rng.normal(0, 0.03, 10)
    for s in range(10):
        _run(tmp_path, "vanilla", 8, s, 0.35 + seed_effect[s])
        _run(tmp_path, "good", 8, s, 0.40 + seed_effect[s] + rng.normal(0, 0.003))
        _run(tmp_path, "same", 8, s, 0.35 + seed_effect[s] + rng.normal(0, 0.003))
        _run(tmp_path, "two_grpo", 2, s, 0.35 + seed_effect[s] + rng.normal(0, 0.003))
    _run(tmp_path, "vanilla", 8, 0, 0.0, stamp="20261001-000000", dirty=True)   # an older, abandoned run
    _run(tmp_path, "base", 8, 0, 0.20)

    out = _load().main(["--results", str(tmp_path), "--reps", "5000"])
    t = out["test"]
    assert set(t) == {"good", "same", "two_grpo"}, "controls are not in the Holm family"
    assert t["good"]["beats_vanilla"] and not t["same"]["beats_vanilla"]
    assert t["two_grpo"]["group_size"] == 2, "a method run at one other group size is compared there"
    vanilla = next(c for c in out["conditions"] if c["method"] == "vanilla")
    assert len(vanilla["accuracies"]) == 10 and min(vanilla["accuracies"]) > 0.2, "the newest run per seed counts"
    assert any("2 runs" in n for n in out["notes"]), "re-runs are listed"
    assert (tmp_path / "aggregate.md").exists() and "| good | 8 | 10 |" in (tmp_path / "aggregate.md").read_text()

    best = _load().main(["--results", str(tmp_path), "--reps", "2000", "--metric", "best"])
    assert best["conditions"][0]["iqm"] != out["conditions"][0]["iqm"]


def test_val_only_runs_are_skipped(tmp_path):
    for s in range(3):
        _run(tmp_path, "vanilla", 8, s, 0.35 + 0.01 * s)
    d = _run(tmp_path, "base", 8, 0, 0.20)
    summary = json.loads((d / "summary.json").read_text())
    summary.update(val_only=True, final={"test_accuracy": None}, best={"test_accuracy": None})
    (d / "summary.json").write_text(json.dumps(summary))
    out = _load().main(["--results", str(tmp_path), "--reps", "500"])
    assert all(c["method"] != "base" for c in out["conditions"]), "a run without test numbers is not a condition"
    assert any("--val-only" in n for n in out["notes"])


def test_passk_is_reported_and_compared_outside_the_falsification_test(tmp_path):
    """pass@k gets its own IQM table and an uncorrected paired difference; the Holm test is unchanged."""
    rng = np.random.default_rng(1)
    for s in range(6):
        e = rng.normal(0, 0.02)
        _run(tmp_path, "vanilla", 8, s, 0.35 + e, passk={"samples": 16, "pass@1": 0.30 + e, "pass@4": 0.55 + e,
                                                          "pass@16": 0.80 + e, "correct_counts": [1, 2]})
        # Higher greedy accuracy, lower pass@16: the "win hiding a loss" pattern.
        _run(tmp_path, "narrow", 8, s, 0.40 + e, passk={"samples": 16, "pass@1": 0.38 + e, "pass@4": 0.52 + e,
                                                         "pass@16": 0.65 + e, "correct_counts": [1, 2]})
        _run(tmp_path, "old_run", 8, s, 0.35 + e)            # no pass@k: an older run, or pass@k off
    out = _load().main(["--results", str(tmp_path), "--reps", "2000"])
    v = next(c for c in out["conditions"] if c["method"] == "vanilla")
    assert set(v["passk"]) == {"pass@1", "pass@4", "pass@16"} and len(v["passk"]["pass@16"]["values"]) == 6
    assert next(c for c in out["conditions"] if c["method"] == "old_run")["passk"] == {}
    d = out["test"]["narrow"]["passk_diff"]
    assert d["pass@16"]["diff"] == pytest.approx(-0.15) and d["pass@16"]["high"] < 0
    assert d["pass@1"]["diff"] > 0
    assert out["test"]["old_run"]["passk_diff"] == {}, "no pass@k on one side: no comparison"
    assert set(out["test"]) == {"narrow", "old_run"}, "pass@k adds no hypotheses to the Holm family"
    text = (tmp_path / "aggregate.md").read_text()
    assert "## Secondary: pass@k" in text and "Exploratory, no correction" in text and "| narrow | -" not in text


def test_runs_with_other_settings_are_flagged(tmp_path):
    """A stray hand-run (other budget) in a condition, or a condition with another shared setup, is noted;
    a method's own lr and advantage parameters are not."""
    import yaml
    from src.config import load_config
    for s in range(3):
        d = _run(tmp_path, "vanilla", 8, s, 0.35 + 0.01 * s)
        extra = ["budget.total_rollouts=6400"] if s == 2 else []
        (d / "config.yaml").write_text(yaml.safe_dump(load_config("vanilla", extra)))
        d = _run(tmp_path, "corpo", 8, s, 0.36)
        (d / "config.yaml").write_text(yaml.safe_dump(load_config("corpo", ["optimizer.lr=3e-5", "advantage.params.c=0.75"])))
        d = _run(tmp_path, "sign", 8, s, 0.36)
        (d / "config.yaml").write_text(yaml.safe_dump(load_config("sign", ["prompt.shots=0"])))
    notes = _load().main(["--results", str(tmp_path), "--reps", "200", "--no-write"])["notes"]
    assert any("vanilla g8 seed 2" in n and "budget.total_rollouts" in n for n in notes)
    assert any(n.startswith("sign g8: shared settings differ") and "prompt.shots" in n for n in notes)
    assert not any(n.startswith("corpo g8") for n in notes), "own lr and parameters are allowed"
    assert not (tmp_path / "aggregate.md").exists(), "--no-write writes nothing"
