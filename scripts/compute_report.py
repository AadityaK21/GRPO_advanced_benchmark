"""After the timing run: measured minutes per run, and the GPU-hours the whole plan needs.

    python scripts/compute_report.py                              # every finished run under results*/
    python scripts/compute_report.py --results results_gate1      # only these folders

Prints the two tables of docs/compute.md, filled in from summary.json (train_seconds, eval_seconds,
gpu_peak_gb, rollouts) and meta.json (gpu, dtype), so nothing is estimated by hand:

1. Time per run: one row per (GPU, dtype, model, method, G), averaged over its finished runs.
2. The grid: every block of configs/grid.yaml x the measured minutes per run, and the tuning sweep of
   configs/tuning.yaml (its trials are shorter: training time scales with rollouts, plus one validation
   check, timed from eval.jsonl).
   GPU-hours are doubled for re-runs, as the plan says.

Minutes per run come from the runs on the GPU named by --gpu (default: the one with the most runs).
A full run's time is taken from runs that used the full budget (budget_rollouts = base.yaml's
total_rollouts); smoke runs are listed but never used for the estimate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finished_runs(folders: list[Path]) -> list[dict]:
    runs = []
    for folder in folders:
        for summary in sorted(folder.glob("*/g*/seed*/summary.json")):
            s = json.loads(summary.read_text(encoding="utf-8"))
            meta_file, cfg_file = summary.parent / "meta.json", summary.parent / "config.yaml"
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
            ev_file = summary.parent / "eval.jsonl"
            checks = [json.loads(x)["seconds"] for x in ev_file.read_text(encoding="utf-8").splitlines() if x.strip()] \
                if ev_file.exists() else []
            runs.append({"dir": summary.parent, "summary": s, "gpu": meta.get("gpu") or "none",
                         "dtype": meta.get("dtype", "?"), "model": (cfg.get("model") or {}).get("name", "?"),
                         "val_check_min": sum(checks) / len(checks) / 60 if checks else None})
    return runs


def time_table(runs: list[dict]) -> tuple[list[str], dict]:
    groups = defaultdict(list)
    for r in runs:
        s = r["summary"]
        groups[(r["gpu"], r["dtype"], r["model"], s["method"], s["group_size"], s.get("budget_rollouts", 0))].append(r)
    lines = ["| GPU | dtype | Model | Method | G | Budget | Runs | Train min | Eval min | Total min | Peak GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        train = sum(r["summary"].get("train_seconds", 0) for r in rs) / len(rs) / 60
        ev = sum(r["summary"].get("eval_seconds", 0) for r in rs) / len(rs) / 60
        peaks = [r["summary"].get("gpu_peak_gb") for r in rs if r["summary"].get("gpu_peak_gb") is not None]
        checks = [r["val_check_min"] for r in rs if r["val_check_min"] is not None]
        rows[key] = {"train_min": train, "eval_min": ev, "total_min": train + ev, "runs": len(rs),
                     "val_check_min": sum(checks) / len(checks) if checks else None}
        gpu, dtype, model, method, g, budget = key
        lines.append(f"| {gpu} | {dtype} | {model.split('/')[-1]} | {method} | {g} | {budget} | {len(rs)} | {train:.1f} | "
                     f"{ev:.1f} | {train + ev:.1f} | {max(peaks) if peaks else '-'} |")
    return lines, rows


def estimate(rows: dict, gpu: str, full_budget: int) -> dict | None:
    """Minutes per full 0.5B training run (train and eval apart) on this GPU, from full-budget runs."""
    trained = [(v, k) for k, v in rows.items() if k[0] == gpu and k[5] == full_budget and k[3] != "base"
               and "1.5B" not in k[2]]
    if not trained:
        return None
    n = sum(v["runs"] for v, _ in trained)
    train = sum(v["train_min"] * v["runs"] for v, _ in trained) / n
    ev = sum(v["eval_min"] * v["runs"] for v, _ in trained) / n
    base = [v for k, v in rows.items() if k[0] == gpu and k[3] == "base" and "1.5B" not in k[2]]
    checks = [v["val_check_min"] for v, _ in trained if v["val_check_min"] is not None]
    return {"train_min": train, "eval_min": ev, "base_min": base[0]["total_min"] if base else ev, "runs": n,
            "val_check_min": sum(checks) / len(checks) if checks else 0.0}


def grid_table(est: dict, scale_1p5b: float) -> tuple[list[str], float]:
    grid_mod, tune_mod = _script("grid"), _script("tune")
    grid = grid_mod.read_grid()
    every = {m: {"overrides": []} for m in tune_mod.all_methods() + ["vanilla"]}   # counts only, no values
    lines = ["| Block | Runs | Minutes / run | GPU-hours | x2 |", "|---|---|---|---|---|"]
    total = 0.0
    for block in grid["blocks"]:
        jobs = grid_mod.jobs_for(grid, block, every)
        minutes = 0.0
        for j in jobs:
            m = est["base_min"] if j["eval_only"] else est["train_min"] + est["eval_min"]
            big = scale_1p5b if any("1.5B" in s for s in j["sets"]) else 1.0
            minutes += m * big
        hours = minutes / 60
        total += hours
        lines.append(f"| {block} | {len(jobs)} | {minutes / max(len(jobs), 1):.1f} | {hours:.0f} | {2 * hours:.0f} |")
    s = tune_mod.read_settings()
    trial_rollouts = next((float(o.split("=")[1]) for o in s["overrides"] if o.startswith("budget.total_rollouts=")), None)
    full = load_config("vanilla")["budget"]["total_rollouts"]
    n_trials = len(tune_mod.all_methods()) * s["trials"] * len(s["seeds"])
    # A trial trains for fewer rollouts and validates once at the end (eval.at_start off): no test, no pass@k.
    per_trial = est["train_min"] * (trial_rollouts or full) / full + est["val_check_min"]
    tune_hours = n_trials * per_trial / 60
    total += tune_hours
    lines.append(f"| tuning ({n_trials} trials) | {n_trials} | {per_trial:.1f} | {tune_hours:.0f} | {2 * tune_hours:.0f} |")
    lines.append(f"| **Total** | | | **{total:.0f}** | **{2 * total:.0f}** |")
    return lines, total


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="*", help="default: results, results_gate1, results_1p5b under the repo")
    p.add_argument("--gpu", help="GPU name to estimate from (default: the one with the most runs)")
    p.add_argument("--scale-1p5b", type=float, default=2.5,
                   help="1.5B minutes / 0.5B minutes, until a 1.5B run is measured (rough: 3x the parameters)")
    args = p.parse_args(argv)
    folders = [Path(f) for f in args.results] if args.results else \
        [ROOT / "results", ROOT / "results_gate1", ROOT / "results_1p5b"]
    runs = finished_runs([f for f in folders if f.exists()])
    if not runs:
        print("no finished runs found: do the timing run first (scripts/run.py --method vanilla --seed 0)")
        return {"runs": 0}
    lines, rows = time_table(runs)
    print("## Time per run (measured)\n\n" + "\n".join(lines))
    gpus = defaultdict(int)
    for r in runs:
        gpus[r["gpu"]] += 1
    gpu = args.gpu or max(gpus, key=gpus.get)
    full = load_config("vanilla")["budget"]["total_rollouts"]
    est = estimate(rows, gpu, full)
    out = {"runs": len(runs), "gpu": gpu, "estimate": est}
    if est is None:
        print(f"\nNo full-budget training run ({full} rollouts) on {gpu}: the grid cannot be estimated yet.")
        return out
    measured_1p5b = [v for k, v in rows.items() if k[0] == gpu and "1.5B" in k[2] and k[5] == full and k[3] != "base"]
    scale = (measured_1p5b[0]["total_min"] / (est["train_min"] + est["eval_min"])) if measured_1p5b else args.scale_1p5b
    g_lines, total = grid_table(est, scale)
    print(f"\n## The grid on {gpu}: {est['train_min']:.1f} min training + {est['eval_min']:.1f} min evaluation per run "
          f"(from {est['runs']} runs); 1.5B = {scale:.2f}x {'(measured)' if measured_1p5b else '(assumed)'}\n")
    print("\n".join(g_lines))
    out["gpu_hours"], out["gpu_hours_x2"] = total, 2 * total
    return out


if __name__ == "__main__":
    main()
