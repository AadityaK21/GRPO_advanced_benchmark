"""End-to-end check of scripts/run.py on the tiny model, offline, on the CPU: a run writes everything it should."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
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
    # Clusters for BV-Blend, made by scripts/make_clusters.py with a stand-in embedding (no download).
    spec = importlib.util.spec_from_file_location("make_clusters", ROOT / "scripts" / "make_clusters.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    questions = [f"t{k} t{k + 1}" for k in range(30)]
    info = mc.make_clusters(questions, k=3, embed=lambda qs: [[float(q.split()[0][1:]) % 3, 0.0] for q in qs])
    (tmp / "clusters.json").write_text(json.dumps(info))
    return tmp


def _args(setup, method, *extra):
    sets = [f"model.name={setup / 'model'}", f"model.tokenizer={setup / 'tokenizer'}", "model.dtype=float32",
            "lora.r=0", "data.val_size=4", "prompt.shots=2", "budget.group_size=2", "budget.rollouts_per_step=4",
            "budget.total_rollouts=8", "budget.micro_batch_size=3", "sampling.max_new_tokens=3",
            "reward.truncated_is_wrong=false",   # 3-token completions are nearly all cut off
            "eval.every=1",
            "eval.val_problems=4", "eval.max_new_tokens=3", "eval.batch_size=3", f"results_dir={setup / 'results'}",
            f"data.clusters_file={setup / 'clusters.json'}"]
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
    k = s["passk"]                                  # base.yaml: 16 samples per test problem, final weights
    assert s["passk_status"] == "done"
    assert k["samples"] == 16 and k["problems"] == 6 and len(k["correct_counts"]) == 6
    assert 0.0 <= k["pass@1"] <= k["pass@4"] <= k["pass@16"] <= 1.0


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


def test_val_only_never_evaluates_the_test_split(setup):
    script = _load_script()
    seen, real = [], script.evaluate

    def spy(model, tokenizer, prompts, *args, **kwargs):
        seen.append(list(prompts))
        return real(model, tokenizer, prompts, *args, **kwargs)

    script.evaluate = spy
    run_dir = script.main(_args(setup, "vanilla", "--eval-only", "--val-only"))
    test_prompts = {f"t{k} t{k + 1}" for k in range(30, 36)}    # the tiny template keeps the question
    assert len(seen) == 1 and len(seen[0]) == 4, "one evaluation: the 4 validation problems"
    assert not set(seen[0]) & test_prompts, "no test problem was evaluated"
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["val_only"] and s["test_problems"] == 0
    assert s["final"]["test_accuracy"] is None and s["best"]["test_accuracy"] is None and s["passk"] is None
    assert s["base_val_accuracy"] == s["best"]["val_accuracy"]
    assert len((run_dir / "val_completions_step0.txt").read_text().split("=====")) == 4
    assert not (run_dir / "ckpt" / "test_completions_final.txt").exists()


def test_without_the_start_check_only_trained_weights_are_validated(setup):
    """eval.at_start=false (tuning trials): no step-0 validation; --eval-only still validates the base model."""
    script = _load_script()
    run_dir = script.main(_args(setup, "vanilla", "--no-ckpt") + ["--set", "eval.at_start=false",
                                                                 "--set", "eval.passk_samples=0"])
    evals = [json.loads(line) for line in (run_dir / "eval.jsonl").read_text().splitlines()]
    s = json.loads((run_dir / "summary.json").read_text())
    assert [e["step"] for e in evals] == [1, 2] and s["base_val_accuracy"] is None
    assert s["best"]["step"] in (1, 2) and s["passk"] is None, "passk_samples 0 switches pass@k off"
    base = script.main(_args(setup, "vanilla", "--eval-only") + ["--set", "eval.at_start=false"])
    assert json.loads((base / "summary.json").read_text())["base_val_accuracy"] is not None


def test_a_crash_during_pass_at_k_keeps_the_finished_run(setup):
    """pass@k runs after summary.json is written: if it dies, the run's primary numbers survive."""
    script = _load_script()
    seen = {}

    def crash(model, *args, **kwargs):
        seen["weights"] = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        raise RuntimeError("CUDA out of memory (simulated)")
    script.evaluate_passk = crash
    with pytest.raises(RuntimeError, match="simulated"):
        script.main(_args(setup, "vanilla", "--no-ckpt") + ["--set", "eval.every=100"])
    run_dir = max((setup / "results" / "vanilla" / "g2").glob("seed0-*"), key=lambda p: p.name)
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["passk_status"] == "pending" and s["passk"] is None
    assert s["final"]["test_accuracy"] is not None and s["steps"] == 2


def test_pass_at_k_uses_the_final_weights_even_when_best_is_earlier(setup):
    script = _load_script()
    seen = {}
    real = script.evaluate_passk

    def spy(model, *args, **kwargs):
        seen["weights"] = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        return real(model, *args, **kwargs)
    script.evaluate_passk = spy
    saved = {}
    real_save = script._save
    script._save = lambda model, path: saved.setdefault(path.name, {n: p.detach().clone()
                                                                     for n, p in model.named_parameters() if p.requires_grad})
    run_dir = script.main(_args(setup, "vanilla") + ["--set", "eval.passk_samples=2"])
    script._save = real_save
    s = json.loads((run_dir / "summary.json").read_text())
    if s["best"]["step"] == s["steps"]:
        pytest.skip("the best step is the last one in this run: nothing to swap back")
    assert all(torch.equal(seen["weights"][n], saved["final"][n]) for n in saved["final"]), "pass@k on the final weights"
    assert not all(torch.equal(seen["weights"][n], saved["best"][n]) for n in saved["best"])


def test_evaluation_logs_one_line_per_batch(setup):
    from src.eval.accuracy import evaluate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(setup / "model")
    tokenizer = AutoTokenizer.from_pretrained(setup / "tokenizer")
    lines = []
    prompts = [f"t{k} t{k + 1}" for k in range(5)]
    res = evaluate(model, tokenizer, prompts, ["#### 1"] * 5, max_new_tokens=2, batch_size=2, log=lines.append)
    assert len(lines) == 3 and "3/3" in lines[-1] and "5/5 problems" in lines[-1]
    assert res == evaluate(model, tokenizer, prompts, ["#### 1"] * 5, max_new_tokens=2, batch_size=2), \
        "logging does not change the result"


@pytest.mark.parametrize("method", ["dapo", "rloo", "sign", "ngrpo", "softmax_grpo", "corpo", "mc_grpo", "two_grpo",
                                    "ebpo", "krpo", "bv_blend", "ha_dw", "signbalance", "lens"])
def test_every_method_runs_end_to_end(setup, method):
    run_dir = _load_script().main(_args(setup, method, "--no-ckpt"))
    s = json.loads((run_dir / "summary.json").read_text())
    train = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
    if method == "mc_grpo":
        # One extra rollout per prompt, paid for out of the budget: 4 per step fits 1 prompt x (2 + 1).
        assert [r["rollouts_total"] for r in train] == [3, 6] and s["rollouts"] == 6
    elif method == "dapo":
        # Refills count too, and the run never passes its budget of 8.
        assert s["rollouts"] <= 8 and all("groups_dropped" in r for r in train)
        assert s["steps"] == len(train)
    else:
        assert s["rollouts"] == 8
    assert all(r["grad_norm"] == r["grad_norm"] for r in train), "no NaN"


def test_two_grpo_refuses_other_group_sizes(setup):
    argv = _args(setup, "two_grpo") + ["--set", "budget.group_size=4", "--set", "budget.rollouts_per_step=8"]
    with pytest.raises(ValueError, match="budget.group_size=2"):
        _load_script().main(argv)


def test_dapo_spends_its_whole_budget_on_mixed_groups(setup):
    argv = _args(setup, "dapo", "--no-ckpt") + ["--set", "budget.total_rollouts=40"]
    run_dir = _load_script().main(argv)
    s = json.loads((run_dir / "summary.json").read_text())
    train = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
    assert 40 - 4 < s["rollouts"] <= 40, "it stops only when the next step would not fit"
    assert s["rollouts"] == train[-1]["rollouts_total"] == sum(r["rollouts"] for r in train)
    assert sum(r["groups_dropped"] for r in train) > 0, "the tiny model gives some uniform groups"
    assert sum(r["refills"] for r in train) > 0
    lrs = [r["lr"] for r in train]
    assert lrs == sorted(lrs, reverse=True) and lrs[0] > lrs[-1], "lr decays with the rollouts spent"


def test_smoke_all_runs_every_method_and_reports_one_table(setup, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("smoke_all", ROOT / "scripts" / "smoke_all.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    monkeypatch.setattr(smoke, "_load_run", _load_script)        # the easy reward, so updates happen
    argv = _args(setup, "vanilla")
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
    out = tmp_path / "table.md"
    rows = smoke.main(["--no-smoke", "--out", str(out), "--local-data", str(setup / "train.jsonl"),
                       str(setup / "test.jsonl")] + [x for s in sets for x in ("--set", s)])
    assert [r["method"] for r in rows] == smoke.all_methods() and len(rows) >= 10
    bad = [(r["method"], r.get("error") or r["problems"]) for r in rows if not r.get("ok")]
    assert not bad, bad
    text = out.read_text()
    assert text.count("\n") == len(rows) + 2 and "| two_grpo | yes | 2 |" in text


def test_smoke_all_reports_a_crash_and_carries_on(setup, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("smoke_all", ROOT / "scripts" / "smoke_all.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    monkeypatch.setattr(smoke, "_load_run", _load_script)
    argv = _args(setup, "vanilla")
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
    rows = smoke.main(["--no-smoke", "--out", str(tmp_path / "t.md"), "--methods", "no_such_method", "vanilla",
                       "--local-data", str(setup / "train.jsonl"), str(setup / "test.jsonl")]
                      + [x for s in sets for x in ("--set", s)])
    assert "error" in rows[0] and rows[1]["ok"]
    assert "CRASH" in (tmp_path / "t.md").read_text()



def test_make_clusters_records_what_it_did(setup):
    info = json.loads((setup / "clusters.json").read_text())
    assert info["cluster_ids"][:6] == [info["cluster_ids"][k] for k in (0, 1, 2, 3, 4, 5)]
    assert len(set(info["cluster_ids"])) == 3 and sum(info["sizes"]) == 30 == info["n"]
    # Points 0, 3, 6, ... share an embedding, so they share a cluster.
    ids = info["cluster_ids"]
    assert len({ids[k] for k in range(0, 30, 3)}) == 1


def test_a_clusters_file_from_other_questions_is_refused(setup, tmp_path):
    bad = json.loads((setup / "clusters.json").read_text())
    bad["questions_sha1"] = "0" * 40
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    argv = _args(setup, "bv_blend") + ["--set", f"data.clusters_file={tmp_path / 'bad.json'}"]
    with pytest.raises(ValueError, match="other questions"):
        _load_script().main(argv)
