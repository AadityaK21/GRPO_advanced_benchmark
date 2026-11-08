"""scripts/tune.py: the plan is equal and spread, trials never see the test split, and collect picks honestly."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import yaml

from test_run_script import _args, _load_script, setup  # noqa: F401  (setup is a fixture)

ROOT = Path(__file__).resolve().parents[1]


def _tune():
    spec = importlib.util.spec_from_file_location("tune", ROOT / "scripts" / "tune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings(**changes):
    s = _tune().read_settings()
    s.update(changes)
    return s


def test_the_real_settings_give_every_method_the_same_budget():
    tune = _tune()
    s = tune.read_settings()
    methods = tune.all_methods()
    assert "vanilla_null" not in methods and "vanilla" in methods and len(methods) == 16
    plan = tune.make_plan(s, methods)
    assert {len(t) for t in plan.values()} == {s["trials"]}, "same number of trials for every method"
    assert all(t["group_size"] == 8 for m, ts in plan.items() if m != "two_grpo" for t in ts)
    assert all(t["group_size"] == 2 for t in plan["two_grpo"])
    assert plan == tune.make_plan(s, methods), "the plan is fixed in advance"
    assert all(o.startswith("budget.total_rollouts") or o.startswith("eval.") for o in s["overrides"])


def test_one_lr_per_stratum_and_parameter_values_spread_over_the_range():
    tune = _tune()
    s = _settings(trials=6)
    plan = tune.make_plan(s, ["vanilla", "corpo", "softmax_grpo"])
    lo, hi = math.log10(s["lr"]["low"]), math.log10(s["lr"]["high"])
    width = (hi - lo) / 6
    for method, trials in plan.items():
        for t in trials:
            x = math.log10(t["lr"])
            assert lo + t["stratum"] * width - 1e-3 <= x <= lo + (t["stratum"] + 1) * width + 1e-3, (method, t)
        assert sorted(t["stratum"] for t in trials) == list(range(6))
    taus = {t["stratum"]: t["params"]["tau"] for t in plan["softmax_grpo"]}
    assert [taus[k] for k in range(6)] == [0.1, 0.3, 0.1, 0.3, 0.1, 0.3], "each tau meets low and high lrs"
    cs = [t["params"]["c"] for t in plan["corpo"]]
    assert sorted(cs) == [0.25, 0.25, 0.5, 0.5, 0.75, 0.75]
    assert any("advantage.params.tau=" in o for o in plan["softmax_grpo"][0]["overrides"])
    assert plan["vanilla"][0]["params"] == {}
    assert [t["lr"] for t in plan["vanilla"]] != [t["lr"] for t in plan["corpo"]], "each method its own draws"


def test_a_per_method_range_is_used_when_given():
    tune = _tune()
    s = _settings(trials=4, lr_ranges={"krpo": {"low": 1e-8, "high": 1e-7}})
    plan = tune.make_plan(s, ["krpo", "vanilla"])
    assert all(1e-8 <= t["lr"] <= 1e-7 for t in plan["krpo"])
    assert all(t["lr"] >= 1e-6 for t in plan["vanilla"])


def test_trials_run_on_validation_only_resume_and_collect(setup, tmp_path):  # noqa: F811
    tune = _tune()
    run = _load_script()
    seen, real = [], run.evaluate
    run.evaluate = lambda model, tok, prompts, *a, **k: (seen.append(list(prompts)), real(model, tok, prompts, *a, **k))[1]
    run.evaluate_passk = lambda *a, **k: pytest.fail("a tuning trial computed pass@k on the test split")
    argv = _args(setup, "vanilla")
    machine = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
    s = _settings(trials=3, seeds=[5], group_size=2, overrides=["budget.total_rollouts=3200", "eval.at_start=false",
                                                                 "eval.every=1000000"])
    root = tmp_path / "tuning"
    local = [str(setup / "train.jsonl"), str(setup / "test.jsonl")]
    dirs = tune.cmd_run(s, ["vanilla", "softmax_grpo"], root, extra_sets=machine, local_data=local, run_module=run)
    assert len(dirs) == 6
    test_prompts = {f"t{k} t{k + 1}" for k in range(30, 36)}
    assert seen and not any(set(p) & test_prompts for p in seen), "no test problem was ever evaluated"
    assert all(len(p) == 4 for p in seen), "only the final validation check (eval.at_start off)"
    for d in dirs:
        summ = json.loads((d / "summary.json").read_text())
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        assert summ["val_only"] and summ["test_problems"] == 0 and summ["steps"] == 2
        assert cfg["budget"]["total_rollouts"] == 8, "the machine's settings come after the tuning defaults"
        trial = int(d.parents[2].name[1:3])
        planned = tune.make_plan(s, [cfg["method"]])[cfg["method"]][trial]
        assert cfg["optimizer"]["lr"] == pytest.approx(planned["lr"]), "the trial's own lr wins"
        if cfg["method"] == "softmax_grpo":
            assert cfg["advantage"]["params"]["tau"] == planned["params"]["tau"]
    assert tune.cmd_run(s, ["vanilla", "softmax_grpo"], root, extra_sets=machine, local_data=local,
                        run_module=run) == [], "finished trials are not run again"

    out = tmp_path / "tuned.yaml"
    res = tune.cmd_collect(s, ["vanilla", "softmax_grpo"], root, out, extra_sets=machine)
    tuned = yaml.safe_load(out.read_text())["methods"]
    assert set(tuned) == {"vanilla", "softmax_grpo"}
    for method, info in tuned.items():
        scores = [tune.score_of(tune.finished(root, method, t, 5, tune.expected_config(s, method, t, machine)), "final")
                  for t in tune.make_plan(s, [method])[method]]
        assert info["val_accuracy"] == max(scores)
        assert info["overrides"][0].startswith("optimizer.lr=")
        assert not any(o.startswith("budget.group_size") for o in info["overrides"]), "G comes from the grid"
    assert any(o.startswith("advantage.params.tau=") for o in tuned["softmax_grpo"]["overrides"])
    assert (root / "tuning.md").exists() and "**chosen**" in (root / "tuning.md").read_text()
    assert isinstance(res["problems"], list)


def _fake_trials(tune, s, root, method, scores):
    """Finished trial folders with the given validation accuracies (trial i -> scores[i])."""
    for t, score in zip(tune.make_plan(s, [method])[method], scores):
        if score is None:
            continue
        exp = tune.expected_config(s, method, t)
        d = tune.trial_dir(root, t, exp) / method / f"g{t['group_size']}" / "seed5-20261108-000000"
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(yaml.safe_dump(exp))
        (d / "summary.json").write_text(json.dumps({"val_only": True, "test_problems": 0,
                                                    "final": {"val_accuracy": score}, "best": {"val_accuracy": 0.9}}))


def test_collect_breaks_ties_low_warns_at_the_edge_and_refuses_gaps(tmp_path):
    tune = _tune()
    s = _settings(trials=4, seeds=[5])
    root = tmp_path / "t"
    _fake_trials(tune, s, root, "vanilla", [0.1, 0.4, 0.4, 0.2])     # tie between strata 1 and 2
    _fake_trials(tune, s, root, "dr_grpo", [0.1, 0.2, 0.3, 0.5])     # best at the top edge
    _fake_trials(tune, s, root, "rloo", [0.3, None, 0.1, 0.1])       # one trial missing
    res = tune.cmd_collect(s, ["vanilla", "dr_grpo", "rloo"], root, tmp_path / "tuned.yaml")
    t = res["tuned"]
    assert t["vanilla"]["trial"] == 1 and not t["vanilla"]["at_edge"], "a tie goes to the lower lr"
    assert t["dr_grpo"]["at_edge"] and any("dr_grpo" in p and "highest" in p for p in res["problems"])
    assert "rloo" not in t and any("rloo" in p and "not finished" in p for p in res["problems"])
    best = tune.cmd_collect(_settings(trials=4, seeds=[5], score="best"), ["vanilla"], root, tmp_path / "b.yaml")
    assert best["tuned"]["vanilla"]["val_accuracy"] == 0.9, "score: best reads the best validation check"
    partial = tune.cmd_collect(s, ["rloo"], root, tmp_path / "p.yaml", allow_incomplete=True)
    assert partial["tuned"]["rloo"]["trial"] == 0 and not partial["tuned"]["rloo"]["complete"]
    assert not (tmp_path / "p.yaml").exists() and partial["written"] == str(root / "tuned_partial.yaml"), \
        "values from an unequal budget never reach the file the grid reads"


def test_changed_settings_never_reuse_old_trials(tmp_path):
    """Widening a method's lr range (HANDOFF step 10) or changing the trial length re-runs its trials."""
    tune = _tune()
    s = _settings(trials=4, seeds=[5])
    root = tmp_path / "t"
    _fake_trials(tune, s, root, "krpo", [0.1, 0.2, 0.3, 0.4])
    assert tune.cmd_collect(s, ["krpo"], root, tmp_path / "a.yaml")["tuned"]["krpo"]["val_accuracy"] == 0.4
    wider = _settings(trials=4, seeds=[5], lr_ranges={"krpo": {"low": 1e-8, "high": 1e-6}})
    res = tune.cmd_collect(wider, ["krpo"], root, tmp_path / "b.yaml")
    assert "krpo" not in res["tuned"] and any("not finished" in p for p in res["problems"]), \
        "old scores are never paired with new lrs"
    shorter = _settings(trials=4, seeds=[5], overrides=["budget.total_rollouts=1600", "eval.at_start=false"])
    assert "krpo" not in tune.cmd_collect(shorter, ["krpo"], root, tmp_path / "c.yaml")["tuned"]
    # A run whose config was edited by hand after the fact is not counted either.
    t = tune.make_plan(s, ["krpo"])["krpo"][3]
    d = next((tune.trial_dir(root, t, tune.expected_config(s, "krpo", t)) / "krpo" / "g8").glob("seed5-*"))
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    cfg["optimizer"]["lr"] = 1.0
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert tune.finished(root, "krpo", t, 5, tune.expected_config(s, "krpo", t)) is None


def test_collect_keeps_other_methods_and_a_relative_root_is_under_the_repo(tmp_path, monkeypatch):
    tune = _tune()
    s = _settings(trials=2, seeds=[5])
    root = tmp_path / "t"
    _fake_trials(tune, s, root, "vanilla", [0.1, 0.2])
    _fake_trials(tune, s, root, "krpo", [0.3, 0.1])
    out = tmp_path / "tuned.yaml"
    tune.cmd_collect(s, ["vanilla"], root, out)
    tune.cmd_collect(s, ["krpo"], root, out)
    assert set(yaml.safe_load(out.read_text())["methods"]) == {"vanilla", "krpo"}, "collect --methods krpo keeps vanilla"
    other = _settings(trials=2, seeds=[6])
    res = tune.cmd_collect(other, ["krpo"], root, out)
    assert res["tuned"] == {} and set(yaml.safe_load(out.read_text())["methods"]) == {"vanilla", "krpo"}
    seen = {}
    monkeypatch.setattr(tune, "cmd_plan", lambda settings, methods, root, extra=(): seen.setdefault("root", root))
    monkeypatch.chdir(tmp_path)
    tune.main(["plan", "--methods", "vanilla", "--root", "runs/tuning"])
    assert seen["root"] == ROOT / "runs" / "tuning", "the same folder run.py writes to, whatever the current folder"


def test_plan_command_prints_and_records(tmp_path, capsys):
    tune = _tune()
    tune.main(["plan", "--methods", "vanilla", "corpo", "--root", str(tmp_path)])
    text = capsys.readouterr().out
    assert "corpo" in text and "2 methods x 8 trials x 1 seeds = 16 runs" in text
    saved = json.loads((tmp_path / "plan.json").read_text())
    assert set(saved["plan"]) == {"vanilla", "corpo"}
