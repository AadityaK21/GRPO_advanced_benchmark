"""Smoke test for scripts/train_grpo.py: two steps end to end on the tiny model, offline, on the CPU."""
import importlib.util
import json
from pathlib import Path

import pytest

from tiny import make_model, make_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("train_grpo", ROOT / "scripts" / "train_grpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_dir(tmp_path):
    model, tokenizer = make_model(), make_tokenizer()
    # The real model has a chat template; give the tiny tokenizer a minimal one. It keeps only the
    # question's first two words, because the tiny vocabulary has no words for the instruction.
    tokenizer.chat_template = "{{ messages[0]['content'].split(' ')[:2] | join(' ') }}"
    model.save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "tokenizer")   # its own folder: next to a Qwen2 config.json,
                                                        # AutoTokenizer would rebuild it as Qwen's tokenizer
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"question": f"t{k} t{k + 1}", "answer": f"#### {k}"}) for k in range(5)))
    return tmp_path


@pytest.mark.parametrize("extra", [[], ["--beta", "0.04"], ["--lora-r", "4", "--beta", "0.04"], ["--num-updates", "2"],
                                   ["--no-grad-ckpt"]])
def test_two_steps_run_and_log(tiny_dir, extra):
    if "--lora-r" in extra:
        pytest.importorskip("peft")
    script = _load_script()
    out = tiny_dir / "run"
    history = script.main(["--model", str(tiny_dir / "model"), "--tokenizer", str(tiny_dir / "tokenizer"),
                           "--data", str(tiny_dir / "data.jsonl"),
                           "--steps", "2", "--group-size", "4", "--max-new-tokens", "4", "--lr", "1e-3",
                           "--out", str(out), *extra])
    lines = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert len(history) == len(lines) == 2
    assert {"reward", "clip_frac", "kl", "loss", "completion_length", "lr", "step"} <= set(lines[0])
    assert lines[0]["lr"] == pytest.approx(1e-3) and lines[1]["lr"] == pytest.approx(5e-4), "linear decay to 0"
    assert json.loads((out / "config.json").read_text())["instruction"]


def test_adapter_off_scores_like_the_base_model_in_chunks():
    """The LoRA reference model must give the base model's log-probs, chunked or not."""
    peft = pytest.importorskip("peft")
    import torch
    from src.train.logprobs import per_token_logps

    script = _load_script()
    base = make_model()
    input_ids = torch.tensor([[0, 5, 6, 7, 8, 9, 10], [3, 4, 5, 6, 7, 8, 9]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    with torch.no_grad():
        want = per_token_logps(base, input_ids, attention_mask, 3, chunk_size=None)

    model = peft.get_peft_model(make_model(), peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear",
                                                               init_lora_weights=False))   # a non-zero adapter
    ref = script.AdapterOff(model)
    with torch.no_grad():
        on = per_token_logps(model, input_ids, attention_mask, 3, chunk_size=2)
        for chunk_size in (None, 2):
            torch.testing.assert_close(per_token_logps(ref, input_ids, attention_mask, 3, chunk_size=chunk_size), want)
    assert not torch.allclose(on, want), "the adapter is on in the model itself"
