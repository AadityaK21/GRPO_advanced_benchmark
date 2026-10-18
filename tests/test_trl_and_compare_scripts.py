"""Smoke tests for scripts/train_trl.py and scripts/compare_runs.py: a few steps on the tiny model, offline."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tiny import make_model, make_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_dir(tmp_path):
    make_model().save_pretrained(tmp_path / "model")
    tokenizer = make_tokenizer()
    tokenizer.chat_template = "{{ messages[0]['content'].split(' ')[:2] | join(' ') }}"
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    (tmp_path / "data.jsonl").write_text(
        "\n".join(json.dumps({"question": f"t{k} t{k + 1}", "answer": f"#### {k}"}) for k in range(5)))
    return tmp_path


def _flags(tiny_dir, out, *extra):
    return ["--model", str(tiny_dir / "model"), "--tokenizer", str(tiny_dir / "tokenizer"),
            "--data", str(tiny_dir / "data.jsonl"), "--steps", "3", "--group-size", "4",
            "--max-new-tokens", "4", "--lr", "1e-3", "--out", str(out), *extra]


@pytest.mark.parametrize("extra", [[], ["--num-updates", "2", "--loss-type", "grpo"]])
def test_trl_script_writes_the_same_log_fields(tiny_dir, monkeypatch, extra):
    pytest.importorskip("trl")
    import trl.trainer.utils as trl_utils
    if not (torch.cuda.is_available() and trl_utils._ChunkedLogProbFunction is not None):
        from test_against_trl import _TorchLogProbs   # TRL's log-prob kernel needs a GPU with Triton
        monkeypatch.setattr(trl_utils, "_ChunkedLogProbFunction", _TorchLogProbs)
    out = tiny_dir / "trl"
    _load("train_trl").main(_flags(tiny_dir, out, *extra))
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2], "one line per sampled batch, numbered like our loop"
    assert {"reward", "completion_length", "truncated_frac", "clip_frac", "kl", "loss", "lr"} <= set(rows[0])


def test_compare_runs_makes_a_plot(tiny_dir):
    pytest.importorskip("matplotlib")
    ours = _load("train_grpo")
    for seed in (0, 1):
        ours.main(_flags(tiny_dir, tiny_dir / "runs" / f"ours_seed{seed}", "--seed", str(seed)))
        ours.main(_flags(tiny_dir, tiny_dir / "runs" / f"trl_seed{seed}", "--seed", str(seed)))   # stand-in
    png = tiny_dir / "runs" / "compare.png"
    _load("compare_runs").main(["--ours", str(tiny_dir / "runs" / "ours_seed*"),
                                "--trl", str(tiny_dir / "runs" / "trl_seed*"), "--window", "2", "--out", str(png)])
    assert png.exists() and png.stat().st_size > 10_000
