"""scripts/compute_report.py: measured minutes per run turn into the plan's GPU-hours."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _report():
    spec = importlib.util.spec_from_file_location("compute_report", ROOT / "scripts" / "compute_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(root, method, g, seed, train_s, eval_s, budget, gpu="NVIDIA A100-SXM4-40GB", model="Qwen/Qwen2.5-0.5B-Instruct",
         checks=(60.0, 60.0)):
    d = root / method / f"g{g}" / f"seed{seed}-20261108-000000"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"method": method, "seed": seed, "group_size": g, "rollouts": budget,
                                                "budget_rollouts": budget, "train_seconds": train_s,
                                                "eval_seconds": eval_s, "gpu_peak_gb": 9.5}))
    (d / "meta.json").write_text(json.dumps({"gpu": gpu, "dtype": "torch.bfloat16"}))
    (d / "config.yaml").write_text(yaml.safe_dump({"model": {"name": model}}))
    (d / "eval.jsonl").write_text("\n".join(json.dumps({"step": i, "seconds": s}) for i, s in enumerate(checks)) + "\n")


def test_grid_hours_follow_the_measured_runs(tmp_path, capsys):
    _run(tmp_path, "vanilla", 8, 0, train_s=1800, eval_s=1200, budget=12800)     # 30 + 20 min
    _run(tmp_path, "vanilla", 8, 1, train_s=1800, eval_s=1200, budget=12800)
    _run(tmp_path, "base", 8, 0, train_s=0, eval_s=900, budget=0)                  # 15 min
    _run(tmp_path, "vanilla", 8, 9, train_s=5, eval_s=5, budget=32)                # a smoke run: never used
    _run(tmp_path, "vanilla", 8, 2, train_s=1, eval_s=1, budget=12800, gpu="Tesla T4")
    out = _report().main(["--results", str(tmp_path), "--scale-1p5b", "2"])
    text = capsys.readouterr().out
    assert out["gpu"] == "NVIDIA A100-SXM4-40GB" and out["estimate"]["runs"] == 2, "the GPU with most runs, full budget only"
    assert out["estimate"]["train_min"] == pytest.approx(30) and out["estimate"]["eval_min"] == pytest.approx(20)
    # gate1: 10 x 50 + 15; tier1: 280 x 50 + 15; null 10 x 50; scale_1p5b: 100 x 100 + 30; tier2: 60 x 50.
    # tuning: 16 methods x 8 trials x (30 x 3200 / 12800 + 1 validation minute) = 128 x 8.5.
    expected = (515 + 14015 + 500 + 10030 + 3000 + 128 * 8.5) / 60
    assert out["gpu_hours"] == pytest.approx(expected)
    assert "| tier1 | 281 |" in text and "| tuning (128 trials) | 128 | 8.5 |" in text and "Tesla T4" in text


def test_no_runs_and_no_full_run(tmp_path, capsys):
    assert _report().main(["--results", str(tmp_path)]) == {"runs": 0}
    _run(tmp_path, "vanilla", 8, 9, train_s=5, eval_s=5, budget=32)
    out = _report().main(["--results", str(tmp_path)])
    assert out["estimate"] is None and "cannot be estimated" in capsys.readouterr().out
