"""scripts/grid.py: the job list, what counts as done, claims, failures, and refusing unsafe starts."""
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest
import yaml

from test_run_script import _args, _load_script, setup  # noqa: F401  (setup is a fixture)

ROOT = Path(__file__).resolve().parents[1]
TIER1 = ["vanilla", "dr_grpo", "dapo", "rloo", "sign", "ngrpo", "softmax_grpo", "corpo", "mc_grpo", "two_grpo"]
TIER2 = ["ebpo", "krpo", "bv_blend", "ha_dw", "signbalance", "lens"]


def _grid():
    spec = importlib.util.spec_from_file_location("grid", ROOT / "scripts" / "grid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tuned(methods=TIER1 + TIER2):
    return {m: {"overrides": [f"optimizer.lr={1e-5 * (i + 1):.4g}"]} for i, m in enumerate(methods)}


def test_the_real_grid_has_the_planned_runs():
    grid = _grid()
    g, tuned = grid.read_grid(), _tuned()
    n = {b: len(grid.jobs_for(g, b, tuned)) for b in g["blocks"]}
    assert n == {"gate1": 11, "tier1": 9 * 3 * 10 + 10 + 1, "null_control": 10, "scale_1p5b": 9 * 10 + 10 + 1, "tier2": 60}
    assert n["tier1"] + n["null_control"] + n["tier2"] == 351, "the plan's 'about 370 runs at 0.5B'"
    tier1 = grid.jobs_for(g, "tier1", tuned)
    assert [j["group_size"] for j in tier1[:90]] == [8] * 90, "G 8 first: the falsification test can start early"
    assert all(j["group_size"] == 2 for j in tier1 if j["method"] == "two_grpo")
    null = grid.jobs_for(g, "null_control", tuned)
    assert [j["seed"] for j in null] == list(range(1000, 1010))
    assert all("optimizer.lr=1e-05" in j["sets"] for j in null), "the null control takes vanilla's tuned values"
    gate1 = grid.jobs_for(g, "gate1", {})
    assert all(not any(s.startswith("optimizer.lr") for s in j["sets"]) for j in gate1), "Gate 1: provisional lr"
    assert all(j["results_dir"] == "results_gate1" for j in gate1)
    big = grid.jobs_for(g, "scale_1p5b", tuned)
    assert all("model.name=Qwen/Qwen2.5-1.5B-Instruct" in j["sets"] and j["results_dir"] == "results_1p5b" for j in big)
    with pytest.raises(LookupError, match="no tuned values"):
        grid.jobs_for(g, "tier1", _tuned(["vanilla"]))
    assert grid.jobs_for(g, "tier1", tuned) == tier1, "a fixed order: an array index always means the same job"


def test_run_argv_for_a_method_and_for_the_base_model():
    grid = _grid()
    g = grid.read_grid()
    jobs = grid.jobs_for(g, "tier1", _tuned())
    j = next(j for j in jobs if j["method"] == "dr_grpo" and j["group_size"] == 4 and j["seed"] == 3)
    argv = grid.run_argv(j)
    assert argv[:4] == ["--method", "dr_grpo", "--seed", "3"] and "budget.group_size=4" in argv
    assert "--eval-only" not in argv and argv[-2:] == ["--set", "results_dir=results"]
    base = next(j for j in jobs if j["eval_only"])
    assert grid.run_argv(base)[:2] == ["--method", "vanilla"] and "--eval-only" in grid.run_argv(base)
    assert base["id"] == "tier1-base-g8-s0"


def _finished(root, method, g, seed, cfg_changes, stamp="20261108-000000"):
    from src.config import load_config
    cfg = load_config("vanilla" if method == "base" else method, [f"budget.group_size={g}"] + cfg_changes)
    d = root / "results" / method / f"g{g}" / f"seed{seed}-{stamp}"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    (d / "summary.json").write_text("{}")
    return d


def test_done_means_finished_with_the_same_values(tmp_path):
    grid = _grid()
    job = {"method": "corpo", "group_size": 8, "seed": 2, "eval_only": False, "results_dir": "results",
           "sets": ["budget.group_size=8", "optimizer.lr=3.162e-05", "advantage.params.c=0.25", "results_dir=results"]}
    assert grid.job_state(job, tmp_path) == ("todo", None)
    old = _finished(tmp_path, "corpo", 8, 2, ["optimizer.lr=1e-05", "advantage.params.c=0.25"], stamp="20261101-000000")
    assert grid.job_state(job, tmp_path) == ("stale", old), "finished with another lr: run again"
    new = _finished(tmp_path, "corpo", 8, 2, ["optimizer.lr=3.162e-05", "advantage.params.c=0.25"])
    assert grid.job_state(job, tmp_path) == ("done", new)
    (new.parent / "seed2-20261109-000000").mkdir()           # an unfinished later run does not count
    assert grid.job_state(job, tmp_path) == ("done", new)


def _small_grid(tmp_path, setup):  # noqa: F811
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump({"seeds": [0, 1], "blocks": {"b": {"use_tuned": False,
                                                                     "results_dir": str(tmp_path / "results"), "entries": [
        {"method": "vanilla", "group_sizes": [2]}, {"method": "base", "eval_only": True, "seeds": [0], "group_sizes": [2]}]}}}))
    argv = _args(setup, "vanilla")
    machine = [argv[i + 1] for i, a in enumerate(argv) if a == "--set" and not argv[i + 1].startswith("results_dir")]
    common = ["--grid", str(path), "--tuned", str(tmp_path / "none.yaml"),
              "--local-data", str(setup / "train.jsonl"), str(setup / "test.jsonl")]
    for s in machine + ["eval.passk_samples=2"]:
        common += ["--set", s]
    return common


def _in_process_runner(calls):
    run = _load_script()

    def runner(argv):
        calls.append(argv)
        run.main(argv + ["--no-ckpt"])
        return 0
    return runner


def test_worker_runs_what_is_left_and_then_nothing(setup, tmp_path):  # noqa: F811
    grid = _grid()
    common = _small_grid(tmp_path, setup)
    calls = []
    counts = grid.main(["worker", "--allow-dirty"] + common, runner=_in_process_runner(calls), root=tmp_path)
    assert counts == {"ran": 3, "failed": 0, "skipped": 0} and len(calls) == 3
    assert sorted(p.parent.name for p in (tmp_path / "results").glob("*/g2")) == ["base", "vanilla"]
    assert not list((tmp_path / "results" / ".claims").glob("*.json")), "claims are released"
    again = grid.main(["worker", "--allow-dirty"] + common, runner=_in_process_runner(calls), root=tmp_path)
    assert again == {"ran": 0, "failed": 0, "skipped": 3} and len(calls) == 3, "finished jobs never run twice"
    st = grid.main(["status"] + common, root=tmp_path)
    assert st["b"] == {"jobs": 3, "done": 3, "stale": 0, "running": 0, "todo": 0, "failed_before": 0}
    lines = grid.main(["list", "--pending", "--write", str(tmp_path / "todo.txt")] + common, root=tmp_path)
    assert lines == [] and (tmp_path / "todo.txt").read_text() == "\n"


def test_claims_failures_and_the_time_limit(setup, tmp_path):  # noqa: F811
    grid = _grid()
    common = _small_grid(tmp_path, setup)
    g = grid.read_grid(Path(common[common.index("--grid") + 1]))
    jobs = grid.jobs_for(g, "b", {})
    # Another worker holds job 0: this one skips it. A claim from a killed job (old) is taken over.
    assert grid.claim(jobs[0], tmp_path, stale_hours=12)
    assert not grid.claim(jobs[0], tmp_path, stale_hours=12), "a claim is exclusive"
    stale = grid.claims_dir(tmp_path) / f"{jobs[1]['id']}.json"
    stale.write_text("{}")
    os.utime(stale, (time.time() - 13 * 3600,) * 2)
    calls = []
    failing = lambda argv: calls.append(argv) or 1                              # noqa: E731
    counts = grid.main(["worker", "--allow-dirty"] + common, runner=failing, root=tmp_path)
    assert counts == {"ran": 0, "failed": 2, "skipped": 1}, "job 0 is someone else's; jobs 1 and 2 fail"
    assert "--seed 1" in " ".join(calls[0]), "the stale claim on job 1 was taken over"
    failed = sorted(p.name for p in (tmp_path / "results" / ".failed").glob("*.txt"))
    assert failed == sorted([f"{jobs[1]['id']}.txt", f"{jobs[2]['id']}.txt"])
    assert grid.claims_dir(tmp_path).joinpath(f"{jobs[0]['id']}.json").exists(), "the other worker's claim stays"
    assert not stale.exists(), "a failed job's claim is released, so it can be retried"
    st = grid.main(["status"] + common, root=tmp_path)
    assert st["b"]["running"] == 1 and st["b"]["failed_before"] == 2 and st["b"]["done"] == 0
    none = grid.main(["worker", "--allow-dirty", "--hours", "0.01", "--minutes-per-run", "60"] + common,
                     runner=failing, root=tmp_path)
    assert none == {"ran": 0, "failed": 0, "skipped": 0}, "no run starts that could pass the time limit"


def test_run_one_runs_exactly_its_index(setup, tmp_path):  # noqa: F811
    grid = _grid()
    common = _small_grid(tmp_path, setup)
    calls = []
    assert grid.main(["run-one", "--block", "b", "--index", "1", "--allow-dirty"] + common,
                     runner=_in_process_runner(calls), root=tmp_path) == "ran"
    assert len(calls) == 1 and calls[0][calls[0].index("--seed") + 1] == "1"
    assert grid.main(["run-one", "--block", "b", "--index", "1", "--allow-dirty"] + common,
                     runner=_in_process_runner(calls), root=tmp_path) == "skipped"
    with pytest.raises(SystemExit, match="outside"):
        grid.main(["run-one", "--block", "b", "--index", "3", "--allow-dirty"] + common, root=tmp_path)


def test_refuses_uncommitted_code(setup, tmp_path, monkeypatch):  # noqa: F811
    grid = _grid()
    common = _small_grid(tmp_path, setup)
    never = lambda argv: pytest.fail("ran from an unsafe tree")                 # noqa: E731
    monkeypatch.setattr(grid, "git_state", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(SystemExit, match="uncommitted"):
        grid.main(["worker"] + common, runner=never, root=tmp_path)
    monkeypatch.setattr(grid, "git_state", lambda: {"commit": None, "dirty": None})
    with pytest.raises(SystemExit, match="not a git checkout"):
        grid.main(["run-one", "--block", "b", "--index", "0"] + common, runner=never, root=tmp_path)
    monkeypatch.setattr(grid, "git_state", lambda: {"commit": "abc", "dirty": False})
    with pytest.raises(SystemExit, match="not committed"):
        grid.check_tree(False, tmp_path / "tuned_not_in_git.yaml")


def test_a_bad_method_fails_before_anything_runs(tmp_path):
    grid = _grid()
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump({"seeds": [0], "blocks": {"b": {"entries": [{"method": "no_such", "group_sizes": [8]}]}}}))
    with pytest.raises(ValueError, match="no method config"):
        grid.main(["list", "--grid", str(path), "--tuned", str(tmp_path / "x.yaml")], root=tmp_path)
    assert json.dumps(grid.read_tuned(tmp_path / "x.yaml")) == "{}"


def test_any_other_setting_makes_a_run_stale(tmp_path):
    """Not only the keys a job sets: a changed base.yaml value, a smoke run or another model never count."""
    grid = _grid()
    g, tuned = grid.read_grid(), _tuned()
    job = next(j for j in grid.jobs_for(g, "tier1", tuned) if j["method"] == "vanilla" and j["seed"] == 0)
    lr = job["sets"][1].split("=")[1]
    _finished(tmp_path, "vanilla", 8, 0, [f"optimizer.lr={lr}", "budget.total_rollouts=6400"], stamp="20261101-000000")
    _finished(tmp_path, "vanilla", 8, 0, [f"optimizer.lr={lr}", "prompt.shots=0"], stamp="20261102-000000")
    assert grid.job_state(job, tmp_path)[0] == "stale"
    good = _finished(tmp_path, "vanilla", 8, 0, [f"optimizer.lr={lr}", "budget.micro_batch_size=4"],
                     stamp="20261103-000000")
    assert grid.job_state(job, tmp_path) == ("done", good), "memory-only settings do not matter"
    base = next(j for j in grid.jobs_for(g, "tier1", tuned) if j["eval_only"])
    d = _finished(tmp_path, "base", 8, 0, ["model.name=Qwen/Qwen2.5-1.5B-Instruct"])
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    cfg["method"] = "base"
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert grid.job_state(base, tmp_path)[0] == "stale", "a 1.5B base run is not the 0.5B base run"
    assert grid.job_state(base, tmp_path, ["model.name=Qwen/Qwen2.5-1.5B-Instruct"])[0] == "done", \
        "machine settings given to the grid are part of what a job runs"


def test_a_partially_tuned_method_is_refused():
    grid = _grid()
    tuned = _tuned()
    tuned["dr_grpo"]["complete"] = False
    with pytest.raises(LookupError, match="fewer trials"):
        grid.jobs_for(grid.read_grid(), "tier1", tuned)


def test_stale_claim_takeover_is_safe_when_workers_race(tmp_path, monkeypatch):
    grid = _grid()
    job = {"id": "b-vanilla-g2-s0"}
    d = grid.claims_dir(tmp_path)
    d.mkdir(parents=True)
    path = d / "b-vanilla-g2-s0.json"
    # (a) the stale claim vanishes between our look and our rename: no crash, no claim (someone else won)
    path.write_text("{}")
    os.utime(path, (time.time() - 13 * 3600,) * 2)
    real_rename = os.rename

    def other_worker_wins(src, dst):
        Path(src).unlink()
        return real_rename(src, dst)
    monkeypatch.setattr(grid.os, "rename", other_worker_wins)
    assert grid.claim(job, tmp_path, stale_hours=12) is False
    monkeypatch.setattr(grid.os, "rename", real_rename)
    # (b) we saw a stale claim, but by the time we rename it another worker has re-claimed: we give it back
    path.write_text(json.dumps({"token": "other-worker"}))
    os.utime(path, (time.time() - 13 * 3600,) * 2)
    real_age = grid._age_hours
    calls = []

    def age(p):
        calls.append(p)
        if len(calls) == 1:
            return 13.0                                # what we saw first
        return real_age(p)
    monkeypatch.setattr(grid, "_age_hours", age)
    os.utime(path, None)                               # the other worker's fresh claim, same file name
    assert grid.claim(job, tmp_path, stale_hours=12) is False
    assert json.loads(path.read_text())["token"] == "other-worker", "the other worker keeps its claim"
    assert not list(d.glob("*.taken-*")), "nothing left behind"
    monkeypatch.setattr(grid, "_age_hours", real_age)
    # Release removes only our own claim; is_claimed survives a vanishing file.
    grid.release(job, tmp_path)
    assert path.exists()
    path.unlink()
    assert grid.is_claimed(job, tmp_path, 12) is False
    assert grid.claim(job, tmp_path, stale_hours=12) and json.loads(path.read_text())["token"] == grid.TOKEN
    grid.release(job, tmp_path)
    assert not path.exists()
