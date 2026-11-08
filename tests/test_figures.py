"""scripts/figures.py on fake finished runs: every figure and table is written, and agrees with aggregate.py."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_results(root: Path, passk: bool = True, base: bool = True, seeds: int = 10) -> None:
    """vanilla and two methods at G 4/8/16, 2-GRPO at 2, the null control, the base model."""
    rng = np.random.default_rng(0)
    effect = {"vanilla": 0.0, "vanilla_null": 0.0, "dr_grpo": 0.04, "sign": 0.0, "two_grpo": -0.02, "dapo": 0.0}
    collapse = {"vanilla": 0.40, "vanilla_null": 0.40, "dr_grpo": 0.38, "sign": 0.41, "two_grpo": 0.55, "dapo": 0.4}
    seed_effect = rng.normal(0, 0.02, seeds)
    runs = [(m, g) for m in ("vanilla", "dr_grpo", "sign") for g in (4, 8, 16)] + [("two_grpo", 2), ("vanilla_null", 8),
                                                                                  ("dapo", 8)]
    for m, g in runs:
        for s in range(seeds):
            seed = s + (1000 if m == "vanilla_null" else 0)
            acc = 0.35 + effect[m] + 0.01 * (g == 16) + seed_effect[s] + rng.normal(0, 0.004)
            d = root / m / f"g{g}" / f"seed{seed}-20261108-000000"
            d.mkdir(parents=True)
            (d / "meta.json").write_text(json.dumps({"gpu": "A100", "dtype": "bf16", "commit": "abc", "dirty": False}))
            pk = {"samples": 16, "pass@1": acc - 0.05, "pass@4": acc + 0.15, "pass@16": acc + 0.35 - effect[m]} if passk else None
            (d / "summary.json").write_text(json.dumps({
                "method": m, "seed": seed, "group_size": g, "rollouts": 12800, "wall_clock_seconds": 3000,
                "final": {"test_accuracy": acc}, "best": {"test_accuracy": acc + 0.005}, "passk": pk}))
            c = collapse[m] + rng.normal(0, 0.02)
            if m == "dapo":       # dynamic sampling: kept groups are mixed (0 uniform); 4 of 12 dropped, then a skipped step
                steps = [{"step": i + 1, "uniform_frac": 0.0, "zero_adv_frac": 0.0, "groups_kept": 8, "groups_dropped": 2}
                         for i in range(4)] + [{"step": 5, "skipped": True, "groups_kept": 0, "groups_dropped": 8}]
            else:
                steps = [{"step": i + 1, "uniform_frac": c + 0.01 * (i % 3), "zero_adv_frac": c * (0.5 if m == "sign" else 1)}
                         for i in range(5)]
            (d / "train.jsonl").write_text("\n".join(json.dumps(r) for r in steps) + "\n")
    if base:
        d = root / "base" / "g8" / "seed0-20261108-000000"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"gpu": "A100", "dtype": "bf16", "commit": "abc", "dirty": False}))
        (d / "summary.json").write_text(json.dumps({
            "method": "base", "seed": 0, "group_size": 8, "rollouts": 0, "wall_clock_seconds": 300,
            "final": {"test_accuracy": 0.20}, "best": {"test_accuracy": 0.20},
            "passk": {"samples": 16, "pass@1": 0.18, "pass@4": 0.45, "pass@16": 0.75} if passk else None}))


def test_every_figure_and_table_from_fake_runs(tmp_path):
    fake_results(tmp_path)
    res = _load("figures").main(["--results", str(tmp_path), "--reps", "1000"])
    out = tmp_path / "figures"
    for stem in ("fig1_accuracy_g8", "fig2_group_sizes", "fig3_collapse", "fig4_passk"):
        for ext in ("pdf", "png", "csv"):
            assert (out / f"{stem}.{ext}").stat().st_size > 0, f"{stem}.{ext}"
    for name in ("table1_g8.md", "table1_g8.tex", "table2_null.md", "table2_null.tex"):
        assert (out / name).exists(), name
    assert not any("skipped" in n for n in res["notes"]), res["notes"]

    agg = _load("aggregate").main(["--results", str(tmp_path), "--reps", "1000", "--no-write"])
    assert not (tmp_path / "aggregate.json").exists(), "figures.py leaves the results table alone"
    rows = list(csv.DictReader(open(out / "fig1_accuracy_g8.csv")))
    assert [r["method"] for r in rows][0] == "vanilla"
    assert {r["method"] for r in rows} == {"vanilla", "dr_grpo", "sign", "two_grpo", "dapo"}
    dr = next(r for r in rows if r["method"] == "dr_grpo")
    cond = next(c for c in agg["conditions"] if c["method"] == "dr_grpo" and c["group_size"] == 8)
    assert float(dr["iqm"]) == pytest.approx(cond["iqm"]), "the figure draws aggregate.py's number"
    assert dr["beats_vanilla"] == "True" and next(r for r in rows if r["method"] == "sign")["beats_vanilla"] == "False"
    assert next(r for r in rows if r["method"] == "two_grpo")["group_size"] == "2"

    fig2 = list(csv.DictReader(open(out / "fig2_group_sizes.csv")))
    assert {r["method"] for r in fig2} == {"dr_grpo", "sign"}, "only methods run at several group sizes"
    fig3 = list(csv.DictReader(open(out / "fig3_collapse.csv")))
    two = next(r for r in fig3 if r["method"] == "two_grpo")
    assert float(two["uniform_frac"]) == pytest.approx(0.56, abs=0.02), "mean over trained steps, skipped ones out"
    assert float(next(r for r in fig3 if r["method"] == "dr_grpo")["gain_iqm"]) == pytest.approx(0.19, abs=0.02)
    assert "vanilla_null" not in {r["method"] for r in fig3} and "r = " in fig3[0]["note"]
    dapo = next(r for r in fig3 if r["method"] == "dapo")
    assert float(dapo["uniform_frac"]) == pytest.approx(16 / 48), "DAPO: dropped groups over all sampled groups"
    assert "| dapo | 8 | 10 |" in (out / "table1_g8.md").read_text() and "| 0.33 |" in (out / "table1_g8.md").read_text()
    fig4 = list(csv.DictReader(open(out / "fig4_passk.csv")))
    assert {r["k"] for r in fig4} == {"pass@1", "pass@4", "pass@16"} and all(r["base"] for r in fig4)
    t1 = (out / "table1_g8.md").read_text()
    assert "| dr_grpo | 8 | 10 |" in t1 and "| yes |" in t1 and "pass@16" in t1
    assert "\\toprule" in (out / "table1_g8.tex").read_text() and "dr\\_grpo" in (out / "table1_g8.tex").read_text()
    assert "20 |" in (out / "table2_null.md").read_text()


def test_missing_pieces_are_skipped_not_crashed(tmp_path):
    fake_results(tmp_path, passk=False, base=False, seeds=3)
    res = _load("figures").main(["--results", str(tmp_path), "--reps", "200"])
    notes = " ".join(res["notes"])
    assert "fig3 skipped" in notes and "fig4 skipped" in notes
    assert (tmp_path / "figures" / "fig1_accuracy_g8.pdf").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _load("figures").main(["--results", str(empty), "--reps", "200"])["written"] == []


def test_figures_never_overwrite_the_results_table(tmp_path):
    fake_results(tmp_path, seeds=4)
    (tmp_path / "aggregate.md").write_text("the 50,000-rep table")
    _load("figures").main(["--results", str(tmp_path), "--reps", "200"])
    assert (tmp_path / "aggregate.md").read_text() == "the 50,000-rep table"


def test_a_method_missing_at_g8_is_not_shown_there(tmp_path):
    """Runs at G 4 and 16 only: aggregate leaves it out of the G 8 test, so figure 1 leaves it out too."""
    fake_results(tmp_path, seeds=4)
    import shutil
    shutil.rmtree(tmp_path / "sign" / "g8")
    _load("figures").main(["--results", str(tmp_path), "--reps", "200"])
    rows = list(csv.DictReader(open(tmp_path / "figures" / "fig1_accuracy_g8.csv")))
    assert "sign" not in {r["method"] for r in rows} and "two_grpo" in {r["method"] for r in rows}
