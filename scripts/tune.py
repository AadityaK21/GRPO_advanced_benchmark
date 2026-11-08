"""The equal-budget tuning sweep: the same number of short trials for every method, scored on validation.

    python scripts/tune.py plan                         # print every trial (and write runs/tuning/plan.json)
    python scripts/tune.py run                          # run every trial not yet finished, all methods
    python scripts/tune.py run --methods dapo corpo     # some methods (e.g. one HPC job per method)
    python scripts/tune.py run --methods dapo --trials 0 3
    python scripts/tune.py collect                      # pick each method's values -> configs/tuned.yaml

Settings: configs/tuning.yaml. Each trial is a `run.py --val-only --no-ckpt` run with its own lr and the
method's own parameters, written to runs/tuning/t<trial>-<fingerprint>/<method>/g<G>/seed<s>-<time>/.
The fingerprint is a hash of the trial's whole resolved config, so a trial counts as finished only if a
run with exactly these settings finished: after widening an lr range, changing the trial length or
base.yaml, the old runs are simply not found and the trials run again. `run` can be stopped and restarted.

`collect` takes, per method, the trial with the highest mean validation accuracy (ties: the lower lr),
updates configs/tuned.yaml (methods not collected now are kept, if they were tuned with the same budget)
and writes runs/tuning/tuning.md. It warns when the chosen lr is in the lowest or highest stratum of its
range. With --allow-incomplete it chooses from finished trials only and writes runs/tuning/tuned_partial.yaml,
never configs/tuned.yaml: an unequal budget must not reach the grid. scripts/grid.py reads
configs/tuned.yaml; vanilla_null uses vanilla's values.

The test split is never evaluated: every trial runs with --val-only, and `run` checks each finished
trial's summary says so.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import differences, fingerprint, load_config  # noqa: E402

TUNING_FILE = ROOT / "configs" / "tuning.yaml"
TUNED_FILE = ROOT / "configs" / "tuned.yaml"
SKIP = {"vanilla_null"}          # the null control is vanilla: it takes vanilla's tuned values
OWN_GROUP_SIZE = {"two_grpo": 2}


def read_settings(path: Path = TUNING_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    s.setdefault("lr_ranges", {})
    s.setdefault("params", {})
    for key in ("trials", "seeds", "plan_seed", "score", "group_size", "overrides", "lr"):
        if key not in s:
            raise KeyError(f"{path}: missing {key!r}")
    if s["score"] not in ("final", "best"):
        raise ValueError(f"{path}: score must be final or best")
    return s


def all_methods(config_dir: Path = ROOT / "configs" / "methods") -> list[str]:
    return sorted(p.stem for p in config_dir.glob("*.yaml") if p.stem not in SKIP)


def make_plan(settings: dict, methods: list[str]) -> dict:
    """{method: [trial, ...]}, each trial {'trial', 'lr', 'stratum', 'params', 'group_size', 'overrides'}.

    lr: log(lr) is split into `trials` equal strata, one trial per stratum, drawn uniformly inside it.
    A parameter with m values gives the trial in stratum s the value number s mod m, so each value gets
    lrs spread over the whole range (tau 0.1 at strata 0, 2, 4, 6; tau 0.3 at 1, 3, 5, 7).
    Deterministic given plan_seed and the method name.
    """
    n = int(settings["trials"])
    plan = {}
    for method in methods:
        rng = random.Random(f"{settings['plan_seed']}-{method}")
        r = settings["lr_ranges"].get(method, settings["lr"])
        lo, hi = math.log10(float(r["low"])), math.log10(float(r["high"]))
        if not hi > lo:
            raise ValueError(f"{method}: lr range must have high > low")
        params = settings["params"].get(method, {})
        g = OWN_GROUP_SIZE.get(method, settings["group_size"])
        trials = []
        for i in range(n):
            s = i                                   # trial i sits in stratum i: trial numbers follow the lr
            log_lr = lo + (s + rng.random()) * (hi - lo) / n
            chosen = {name: values[s % len(values)] for name, values in sorted(params.items())}
            overrides = [f"optimizer.lr={10 ** log_lr:.4g}", f"budget.group_size={g}"]
            overrides += [f"advantage.params.{k}={v}" for k, v in chosen.items()]
            trials.append({"trial": i, "lr": float(f"{10 ** log_lr:.4g}"), "stratum": s, "params": chosen,
                           "group_size": g, "overrides": overrides})
        plan[method] = trials
    return plan


def trial_sets(settings: dict, trial: dict, extra_sets=()) -> list[str]:
    """Shared tuning settings, then the machine's (--set), then the trial's own lr and parameters, which win."""
    return list(settings["overrides"]) + list(extra_sets) + list(trial["overrides"])


def expected_config(settings: dict, method: str, trial: dict, extra_sets=()) -> dict:
    return load_config(method, trial_sets(settings, trial, extra_sets))


def trial_dir(root: Path, trial: dict, expected: dict) -> Path:
    return root / f"t{trial['trial']:02d}-{fingerprint(expected)}"


def finished(root: Path, method: str, trial: dict, seed: int, expected: dict) -> Path | None:
    """The newest finished run of this trial and seed with exactly the expected settings, or None."""
    folder = trial_dir(root, trial, expected) / method / f"g{trial['group_size']}"
    for summary in sorted(folder.glob(f"seed{seed}-*/summary.json"), reverse=True):
        cfg = yaml.safe_load((summary.parent / "config.yaml").read_text(encoding="utf-8"))
        if not differences(cfg, expected):
            return summary.parent
    return None


def trial_argv(settings: dict, method: str, trial: dict, seed: int, root: Path, extra_sets=(), local_data=None) -> list[str]:
    argv = ["--method", method, "--seed", str(seed), "--val-only", "--no-ckpt"]
    sets = trial_sets(settings, trial, extra_sets)
    sets.append(f"results_dir={trial_dir(root, trial, expected_config(settings, method, trial, extra_sets))}")
    for s in sets:
        argv += ["--set", s]
    if local_data:
        argv += ["--local-data", *local_data]
    if "--val-only" not in argv:                       # never remove: tuning must not see the test split
        raise AssertionError("a tuning trial must run with --val-only")
    return argv


def _load_run():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_plan(settings, methods, root: Path, extra_sets=()) -> dict:
    plan = make_plan(settings, methods)
    root.mkdir(parents=True, exist_ok=True)
    (root / "plan.json").write_text(json.dumps({"settings": settings, "plan": plan}, indent=2), encoding="utf-8")
    for method, trials in plan.items():
        for t in trials:
            seeds = settings["seeds"]
            exp = expected_config(settings, method, t, extra_sets)
            done = sum(finished(root, method, t, s, exp) is not None for s in seeds)
            params = ", ".join(f"{k}={v}" for k, v in t["params"].items())
            print(f"{method:13s} t{t['trial']:02d}  lr {t['lr']:.3g}  {params:12s}  G {t['group_size']}  "
                  f"{done}/{len(seeds)} seeds done")
    total = sum(len(t) for t in plan.values()) * len(settings["seeds"])
    print(f"\n{len(plan)} methods x {settings['trials']} trials x {len(settings['seeds'])} seeds = {total} runs")
    return plan


def cmd_run(settings, methods, root: Path, trials=None, extra_sets=(), local_data=None, run_module=None) -> list[Path]:
    plan = cmd_plan(settings, methods, root, extra_sets)
    run = run_module or _load_run()
    out = []
    for method, method_trials in plan.items():
        for t in method_trials:
            if trials is not None and t["trial"] not in trials:
                continue
            exp = expected_config(settings, method, t, extra_sets)
            for seed in settings["seeds"]:
                if finished(root, method, t, seed, exp) is not None:
                    continue
                print(f"\n===== tuning {method} t{t['trial']:02d} seed {seed}: lr {t['lr']:.3g} {t['params']} =====",
                      flush=True)
                run_dir = run.main(trial_argv(settings, method, t, seed, root, extra_sets, local_data))
                s = json.loads((Path(run_dir) / "summary.json").read_text())
                if not s.get("val_only") or s.get("test_problems"):
                    raise RuntimeError(f"{run_dir}: a tuning trial evaluated the test split")
                out.append(Path(run_dir))
    return out


def score_of(run_dir: Path, metric: str) -> float:
    return json.loads((run_dir / "summary.json").read_text())[metric]["val_accuracy"]


def cmd_collect(settings, methods, root: Path, out_file: Path = TUNED_FILE, allow_incomplete: bool = False,
                extra_sets=()) -> dict:
    plan = make_plan(settings, methods)
    n = int(settings["trials"])
    tuned, lines, problems = {}, ["# Tuning results (validation accuracy; test never evaluated)", ""], []
    for method, trials in plan.items():
        rows = []
        for t in trials:
            exp = expected_config(settings, method, t, extra_sets)
            dirs = [finished(root, method, t, s, exp) for s in settings["seeds"]]
            if any(d is None for d in dirs):
                rows.append({**t, "score": None})
                continue
            rows.append({**t, "score": sum(score_of(d, settings["score"]) for d in dirs) / len(dirs),
                         "runs": [str(d) for d in dirs]})
        missing = [r["trial"] for r in rows if r["score"] is None]
        if missing and not allow_incomplete:
            problems.append(f"{method}: trials {missing} not finished")
            continue
        done = [r for r in rows if r["score"] is not None]
        if not done:
            problems.append(f"{method}: no finished trials")
            continue
        best = max(done, key=lambda r: (r["score"], -r["lr"]))       # ties: the lower lr
        at_edge = best["stratum"] in (0, n - 1)
        tuned[method] = {"overrides": [o for o in best["overrides"] if not o.startswith("budget.group_size")],
                         "lr": best["lr"], "params": best["params"], "val_accuracy": best["score"],
                         "trial": best["trial"], "at_edge": at_edge, "complete": not missing}
        if at_edge:
            problems.append(f"{method}: chosen lr {best['lr']:.3g} is in the {'lowest' if best['stratum'] == 0 else 'highest'} "
                            "stratum: widen its range in configs/tuning.yaml and re-run its whole sweep")
        lines += [f"## {method}", "", "| Trial | lr | Params | Val accuracy |", "|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: r["lr"]):
            mark = " **chosen**" if r is best else ""
            score = "not finished" if r["score"] is None else f"{r['score']:.4f}"
            lines.append(f"| t{r['trial']:02d} | {r['lr']:.3g} | {r['params'] or '-'} | {score}{mark} |")
        lines.append("")
    if problems:
        lines += ["## Problems", ""] + [f"- {p}" for p in problems]
    header = {"written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "by": "scripts/tune.py collect", "from": str(root),
              "budget": {"trials": settings["trials"], "seeds": settings["seeds"], "score": settings["score"],
                         "group_size": settings["group_size"], "overrides": settings["overrides"]}}
    text = "\n".join(lines) + "\n"
    print(text)
    root.mkdir(parents=True, exist_ok=True)
    (root / "tuning.md").write_text(text, encoding="utf-8")
    partial = any(not v["complete"] for v in tuned.values())
    target = root / "tuned_partial.yaml" if partial else out_file
    merged = dict(tuned)
    if not partial and out_file.exists():                 # keep methods collected earlier with the same budget
        old = yaml.safe_load(out_file.read_text(encoding="utf-8")) or {}
        if old.get("budget") == header["budget"]:
            merged = {**old.get("methods", {}), **tuned}
        elif old.get("methods"):
            problems.append(f"{out_file} was tuned with another budget: its other methods are dropped")
    if tuned:
        target.write_text("# Written by scripts/tune.py collect. Do not edit by hand: re-run the sweep.\n"
                          + yaml.safe_dump({**header, "methods": merged}, sort_keys=False), encoding="utf-8")
    return {"tuned": tuned, "problems": problems, "written": str(target) if tuned else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["plan", "run", "collect"])
    p.add_argument("--methods", nargs="*", help="default: every method except vanilla_null")
    p.add_argument("--trials", nargs="*", type=int, help="run only these trial numbers")
    p.add_argument("--settings", default=str(TUNING_FILE))
    p.add_argument("--root", default=str(ROOT / "runs" / "tuning"), help="where trials are written")
    p.add_argument("--out", default=str(TUNED_FILE), help="collect: where to write the chosen values")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="extra override for every trial (a machine setting such as a local model path, never a "
                        "tuning choice). It is part of each trial's fingerprint: pass the same --set to plan, run "
                        "and collect")
    p.add_argument("--local-data", nargs=2, metavar=("TRAIN_JSONL", "TEST_JSONL"))
    p.add_argument("--allow-incomplete", action="store_true", help="collect: choose from the finished trials only")
    args = p.parse_args(argv)
    settings = read_settings(Path(args.settings))
    methods = args.methods or all_methods()
    root = Path(args.root)
    root = root if root.is_absolute() else ROOT / root      # run.py writes relative paths under the repo too
    for m in methods:
        load_config(m)                                       # a bad method name fails before anything runs
    if args.command == "plan":
        return cmd_plan(settings, methods, root, args.set)
    if args.command == "run":
        return cmd_run(settings, methods, root, args.trials, args.set, args.local_data)
    return cmd_collect(settings, methods, root, Path(args.out), args.allow_incomplete, args.set)


if __name__ == "__main__":
    main()
