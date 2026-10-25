"""Smoke-test every method on the GPU in one go: each config in configs/methods/ runs `run.py --smoke`.

    python scripts/smoke_all.py                      # all methods, seed 0
    python scripts/smoke_all.py --methods dapo mc_grpo

Prints one table and writes it to runs/smoke/smoke_all.md: did each method run, how many steps and
rollouts it used, the mean training reward and gradient norm, its time and peak GPU memory, and the
checks below. A method that crashes is reported and the rest still run.

Checks per method (a failed check is listed in the last column):
- every logged loss and gradient norm is finite
- the run spent no more rollouts than its budget
- at least one step changed the weights (grad_norm > 0)
- 2-GRPO ran at group size 2; MC-GRPO sampled one extra completion per prompt; DAPO logged its filter

This is a check that the code runs on the real model and GPU, not a result: 2 steps say nothing about
which method is better.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GROUP_SIZE_OVERRIDES = {"two_grpo": ["budget.group_size=2"]}


def _load_run():
    spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def all_methods(config_dir: Path = ROOT / "configs" / "methods") -> list[str]:
    return sorted(p.stem for p in config_dir.glob("*.yaml"))


def check_run(method: str, run_dir: Path) -> dict:
    """Read one run's files and return its row of the table."""
    s = json.loads((run_dir / "summary.json").read_text())
    cfg_text = (run_dir / "config.yaml").read_text()
    import yaml
    cfg = yaml.safe_load(cfg_text)
    train_path = run_dir / "train.jsonl"
    train = [json.loads(line) for line in train_path.read_text().splitlines()] if train_path.exists() else []
    problems = []
    finite = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
    if not all(finite(r.get("loss", 0.0)) and finite(r.get("grad_norm", 0.0)) for r in train if not r.get("skipped")):
        problems.append("non-finite loss or grad")
    if s["rollouts"] > cfg["budget"]["total_rollouts"]:
        problems.append(f"over budget ({s['rollouts']} > {cfg['budget']['total_rollouts']})")
    if not any(r.get("grad_norm", 0) > 0 for r in train):
        problems.append("no step changed the weights")
    g = cfg["budget"]["group_size"]
    if method == "two_grpo" and g != 2:
        problems.append(f"ran at G = {g}")
    if method == "mc_grpo" and train and train[0]["rollouts"] % (g + 1):
        problems.append("rollouts per step not a multiple of G + 1")
    if cfg["filter"]["drop_uniform_groups"] and train and not all("groups_dropped" in r for r in train):
        problems.append("dynamic sampling did not log its filter")
    trained = [r for r in train if not r.get("skipped")]
    mean = lambda key: sum(r[key] for r in trained) / len(trained) if trained else float("nan")
    return {
        "method": method, "ok": not problems, "G": g, "steps": s["steps"], "rollouts": s["rollouts"],
        "budget": cfg["budget"]["total_rollouts"], "reward": mean("reward"), "grad_norm": mean("grad_norm"),
        "dropped": sum(r.get("groups_dropped", 0) for r in train),
        "minutes": round(s["wall_clock_seconds"] / 60, 1), "peak_gb": s.get("gpu_peak_gb"),
        "problems": "; ".join(problems), "run_dir": str(run_dir),
    }


def table(rows: list[dict]) -> str:
    head = "| Method | OK | G | Steps | Rollouts / budget | Reward | Grad norm | Groups dropped | Min | Peak GB | Problems |"
    lines = [head, "|" + "---|" * 11]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['method']} | CRASH | | | | | | | | | {r['error']} |")
            continue
        lines.append(f"| {r['method']} | {'yes' if r['ok'] else 'NO'} | {r['G']} | {r['steps']} | "
                     f"{r['rollouts']} / {r['budget']} | {r['reward']:.3f} | {r['grad_norm']:.3f} | {r['dropped']} | "
                     f"{r['minutes']} | {r['peak_gb'] if r['peak_gb'] is not None else '-'} | {r['problems']} |")
    return "\n".join(lines)


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="*", help="default: every configs/methods/*.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="passed to every run")
    p.add_argument("--local-data", nargs=2, metavar=("TRAIN_JSONL", "TEST_JSONL"))
    p.add_argument("--no-smoke", action="store_true", help="do not add --smoke (tests pass their own tiny settings)")
    p.add_argument("--out", default=None, help="where to write the table (default: runs/smoke/smoke_all.md)")
    args = p.parse_args(argv)

    run = _load_run()
    rows = []
    for method in args.methods or all_methods():
        argv_run = ["--method", method, "--seed", str(args.seed), "--no-ckpt"]
        for item in list(args.set) + GROUP_SIZE_OVERRIDES.get(method, []):
            argv_run += ["--set", item]
        if args.local_data:
            argv_run += ["--local-data", *args.local_data]
        if not args.no_smoke:
            argv_run.append("--smoke")
        print(f"\n===== {method} =====", flush=True)
        t0 = time.perf_counter()
        try:
            rows.append(check_run(method, run.main(argv_run)))
        except Exception as e:                     # report and keep going: one table for every method
            traceback.print_exc()
            rows.append({"method": method, "error": f"{type(e).__name__}: {e}"[:200],
                         "minutes": round((time.perf_counter() - t0) / 60, 1)})
    text = table(rows)
    print("\n" + text, flush=True)
    out = Path(args.out) if args.out else ROOT / "runs" / "smoke" / "smoke_all.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    rows = main()
    sys.exit(0 if all(r.get("ok") for r in rows) else 1)
