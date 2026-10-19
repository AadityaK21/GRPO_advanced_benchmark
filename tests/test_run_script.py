"""End-to-end check of scripts/run.py on the tiny model, offline, on the CPU: a run writes everything it should."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from tiny import make_model, make_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The tiny model almost never writes the right number, so every group would be all-wrong and no
    # gradient would flow. Train on an easy reward instead: the completion's first word is t0..t30.
    module.exact_match_reward = _first_word_below_31
    return module


def _first_word_below_31(texts, answers):
    out = []
    for text in texts:
        words = text.split()
        out.append(1.0 if words and words[0][1:].isdigit() and int(words[0][1:]) <= 30 else 0.0)
    return out


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    model, tokenizer = make_model(), make_tokenizer()
    # The tiny vocabulary has no words for the few-shot examples or the instruction: the template keeps
    # only the first two words of the last turn.
    tokenizer.chat_template = "{{ messages[-1]['content'].split(' ')[:2] | join(' ') }}"
    model.save_pretrained(tmp / "model")
    tokenizer.save_pretrained(tmp / "tokenizer")
    rows = lambda ks: "\n".join(json.dumps({"question": f"t{k} t{k + 1}", "answer": f"#### {k % 7}"}) for k in ks)
    (tmp / "train.jsonl").write_text(rows(range(30)))
    (tmp / "test.jsonl").write_text(rows(range(30, 36)))
    return tmp


def _args(setup, method, *extra):
    sets = [f"model.name={setup / 'model'}", f"model.tokenizer={setup / 'tokenizer'}", "model.dtype=float32",
            "lora.r=0", "data.val_size=4", "prompt.shots=2", "budget.group_size=2", "budget.rollouts_per_step=4",
            "budget.total_rollouts=8", "budget.micro_batch_size=3", "sampling.max_new_tokens=3",
            "reward.truncated_is_wrong=false",   # 3-token completions are nearly all cut off
            "eval.every=1",
            "eval.val_problems=4", "eval.max_new_tokens=3", "eval.batch_size=3", f"results_dir={setup / 'results'}"]
    argv = ["--method", method, "--seed", "0", "--local-data", str(setup / "train.jsonl"), str(setup / "test.jsonl")]
    for s in sets:
        argv += ["--set", s]
    return argv + list(extra)


@pytest.mark.parametrize("method", ["vanilla", "dr_grpo"])
def test_a_run_writes_config_seed_commit_and_results(setup, method):
    run_dir = _load_script().main(_args(setup, method))
    assert run_dir.parent == setup / "results" / method / "g2"
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert cfg["method"] == method and cfg["budget"]["group_size"] == 2, "the resolved config, overrides included"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["seed"] == 0 and "commit" in meta and "dirty" in meta
    assert meta["splits"]["val"] == 4 and len(meta["splits"]["shots"]) == 2 and meta["splits"]["train"] == 24

    train = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in train] == [1, 2] and [r["rollouts_total"] for r in train] == [4, 8]
    assert any(r["grad_norm"] > 0 for r in train), "some group had unequal rewards, so the model was updated"
    for key in ("reward", "zero_adv_frac", "reward_var", "entropy", "grad_norm", "completion_length", "lr"):
        assert key in train[0], key
    evals = [json.loads(line) for line in (run_dir / "eval.jsonl").read_text().splitlines()]
    assert [e["step"] for e in evals] == [0, 1, 2], "validation before training and after every step here"

    s = json.loads((run_dir / "summary.json").read_text())
    assert s["rollouts"] == 8 == s["budget_rollouts"] and s["steps"] == 2
    assert len(s["test_correct_final"]) == 6 == s["test_problems"]
    assert s["final"]["test_accuracy"] == s["test_correct_final"].count("1") / 6
    assert s["best"]["step"] in (0, 1, 2)
    assert s["best"]["val_accuracy"] == max(e["val_accuracy"] for e in evals)
    assert 0.0 <= s["final"]["test_format_rate"] <= 1.0 and "val_format_rate" in evals[0]
    for key in ("uniform_frac", "adv_abs_mean"):
        assert key in train[0], key


def test_seeds_change_the_run_and_repeat_exactly(setup):
    script = _load_script()
    reads = []
    for seed in (0, 0, 1):
        argv = _args(setup, "vanilla")
        argv[argv.index("--seed") + 1] = str(seed)
        run_dir = script.main(argv)
        rows = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
        reads.append(([(r["reward"], r["entropy"], r["grad_norm"]) for r in rows],
                      json.loads((run_dir / "meta.json").read_text())["splits"]["shots"]))
    assert reads[0] == reads[1], "same seed, same run"
    assert reads[0][0] != reads[2][0], "another seed samples differently"
    assert reads[0][1] == reads[2][1], "the few-shot examples do not depend on the run seed"


def test_eval_only_is_the_base_model(setup):
    run_dir = _load_script().main(_args(setup, "vanilla", "--eval-only"))
    assert run_dir.parent.parent.name == "base"
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["steps"] == 0 and s["rollouts"] == 0 and not (run_dir / "train.jsonl").exists()
    assert s["final"]["test_accuracy"] == s["best"]["test_accuracy"]
