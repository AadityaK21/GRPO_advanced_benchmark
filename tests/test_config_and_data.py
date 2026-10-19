"""Checks for src/config.py and src/data.py: matched conditions are enforced, splits are fixed."""
import shutil

import pytest

from src.config import CONFIG_DIR, load_config, parse_override
from src.data import build_messages, clean_solution, make_splits


def test_methods_inherit_everything_shared():
    vanilla, dr = load_config("vanilla"), load_config("dr_grpo")
    assert vanilla["method"] == "vanilla" and dr["method"] == "dr_grpo"
    assert vanilla["advantage"]["name"] == "vanilla" and dr["advantage"]["name"] == "dr_grpo"
    assert vanilla["loss"]["loss_type"] == "grpo" and dr["loss"]["loss_type"] == "dr_grpo"
    for section in ("model", "lora", "optimizer", "data", "prompt", "sampling", "reward", "budget", "eval"):
        assert vanilla[section] == dr[section], f"{section} must be identical across methods"


def test_a_method_file_cannot_change_the_shared_setup(tmp_path):
    shutil.copytree(CONFIG_DIR, tmp_path / "configs")
    (tmp_path / "configs" / "methods" / "greedy.yaml").write_text("advantage:\n  name: vanilla\nbudget:\n  total_rollouts: 99999\n")
    with pytest.raises(ValueError, match="budget"):
        load_config("greedy", config_dir=tmp_path / "configs")
    (tmp_path / "configs" / "methods" / "typo.yaml").write_text("advantage:\n  name: vanilla\nloss:\n  loss_tpye: dapo\n")
    with pytest.raises(KeyError, match="loss_tpye"):
        load_config("typo", config_dir=tmp_path / "configs")


def test_overrides():
    cfg = load_config("vanilla", ["budget.group_size=4", "optimizer.lr=3e-6", "eval.test_problems=null",
                                  "advantage.params.temperature=0.5"])
    assert cfg["budget"]["group_size"] == 4
    assert cfg["optimizer"]["lr"] == pytest.approx(3e-6) and isinstance(cfg["optimizer"]["lr"], float)
    assert cfg["eval"]["test_problems"] is None
    assert cfg["advantage"]["params"] == {"temperature": 0.5}, "a method's own parameters are free-form"
    with pytest.raises(KeyError, match="group_sise"):
        load_config("vanilla", ["budget.group_sise=4"])
    with pytest.raises(ValueError, match="no method config"):
        load_config("not_a_method")
    with pytest.raises(ValueError):
        load_config("vanilla", ["budget.group_size=1"])
    assert parse_override("model.name=Qwen/Qwen2.5-1.5B-Instruct") == ("model.name", "Qwen/Qwen2.5-1.5B-Instruct")


def _rows(n):
    return [{"question": f"q{i}", "answer": f"a <<1+1=2>>2\n#### {i}"} for i in range(n)]


def test_splits_are_fixed_disjoint_and_keep_gsm8k_indices():
    a = make_splits(_rows(100), val_size=20, shots=3, split_seed=1234)
    b = make_splits(_rows(100), val_size=20, shots=3, split_seed=1234)
    assert a == b, "the same split_seed must give the same split on every machine and every run"
    assert [len(a[k]) for k in ("shots", "val", "train")] == [3, 20, 77]
    ids = [r["idx"] for k in ("shots", "val", "train") for r in a[k]]
    assert sorted(ids) == list(range(100)), "every problem in exactly one split"
    assert all(r["question"] == f"q{r['idx']}" for r in a["train"])
    assert make_splits(_rows(100), 20, 3, split_seed=7)["val"] != a["val"]
    assert make_splits(_rows(100), 20, 0, split_seed=1234)["val"] == a["val"], \
        "the validation set must not depend on the number of shots"
    with pytest.raises(ValueError):
        make_splits(_rows(10), val_size=8, shots=2, split_seed=0)


def test_few_shot_prompt():
    assert clean_solution("She has 48/2 = <<48/2=24>>24 left.\n#### 24") == "She has 48/2 = 24 left.\n#### 24"
    msgs = build_messages("Q?", "Think.", [{"question": "S1", "answer": "x <<1=1>>1\n#### 1"},
                                           {"question": "S2", "answer": "#### 2"}])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[0]["content"] == "S1 Think." and msgs[1]["content"] == "x 1\n#### 1"
    assert msgs[-1]["content"] == "Q? Think."
    assert build_messages("Q?", "Think.", []) == [{"role": "user", "content": "Q? Think."}]
