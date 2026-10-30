"""Checks for scripts/aggregate.py on fake finished runs."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("aggregate", ROOT / "scripts" / "aggregate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(results, method, g, seed, acc, stamp="20261101-000000", dirty=False):
    d = results / method / f"g{g}" / f"seed{seed}-{stamp}"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"seed": seed, "gpu": "A100", "dtype": "torch.bfloat16",
                                             "commit": "abc", "dirty": dirty}))
    (d / "summary.json").write_text(json.dumps({
        "method": method, "seed": seed, "group_size": g, "rollouts": 12800, "wall_clock_seconds": 600,
        "final": {"test_accuracy": acc}, "best": {"test_accuracy": acc + 0.01}}))
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
