"""The benchmark grid: one job per (method, group size, seed), resumable, from configs/grid.yaml.

    python scripts/grid.py status                          # done / to do / running, per block
    python scripts/grid.py list --block tier1 --pending    # the run.py commands still to do
    python scripts/grid.py worker --block tier1 --hours 23.5   # run pending jobs one after another
    python scripts/grid.py run-one --block tier1 --index 17    # job 17 only (a PBS / Slurm array task)

A job is done when its results folder has a finished run (summary.json) whose whole resolved config
equals what the job would run now (base.yaml + the method's file + tuned values + the block's settings),
apart from memory-only settings (src/config.py MEMORY_ONLY). A finished run with any other value (from
before re-tuning, a changed base.yaml, a smoke run) is listed as stale and the job runs again;
aggregate.py uses the newest run per seed and notes any seeds or conditions whose settings differ.

Safety:
- Refuses to start from a git tree with uncommitted changes, or with no commit: every result must name
  the code that made it. --allow-dirty is for tests and dry runs only.
- A block with use_tuned refuses any method without values in configs/tuned.yaml (scripts/tune.py collect).
- Workers can share one grid: a job is claimed by creating results/.claims/<job>.json atomically, so two
  workers never run the same job. A claim older than --stale-hours (a killed job) can be taken over: it is
  renamed away first, which only one worker can do, and a worker releases only a claim holding its token.
- A method tuned on fewer trials than planned (`complete: false`) is refused.

Each job runs as its own process (`python scripts/run.py ...`), so GPU memory is freed between runs.
HPC job scripts that call this: hpc/grid_worker.pbs (IIT Delhi, PBS) and hpc/grid_array.slurm.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import differences, load_config  # noqa: E402
from src.runlog import git_state  # noqa: E402

GRID_FILE = ROOT / "configs" / "grid.yaml"
TUNED_FILE = ROOT / "configs" / "tuned.yaml"
TUNED_AS = {"vanilla_null": "vanilla"}          # the null control is vanilla


# ---- building the job list ------------------------------------------------------------------------------

def read_grid(path: Path = GRID_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_tuned(path: Path = TUNED_FILE) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("methods", {})


def jobs_for(grid: dict, block: str, tuned: dict) -> list[dict]:
    """Every job of a block, in a fixed order (an array index always means the same job)."""
    if block not in grid["blocks"]:
        raise KeyError(f"no block {block!r} in the grid; blocks: {list(grid['blocks'])}")
    b = grid["blocks"][block]
    results_dir = b.get("results_dir", "results")
    jobs = []
    for entry in b["entries"]:
        methods = entry.get("methods") or [entry["method"]]
        eval_only = bool(entry.get("eval_only"))
        for g in entry.get("group_sizes", [8]):
            for method in methods:
                for seed in entry.get("seeds", grid["seeds"]):
                    sets = [f"budget.group_size={g}"]
                    if b.get("use_tuned") and not eval_only:
                        name = TUNED_AS.get(method, method)
                        if name not in tuned:
                            raise LookupError(f"block {block}: no tuned values for {name} in configs/tuned.yaml. "
                                              "Run scripts/tune.py, or use a block with use_tuned: false")
                        if tuned[name].get("complete") is False:
                            raise LookupError(f"block {block}: {name} was tuned on fewer trials than planned")
                        sets += list(tuned[name]["overrides"])
                    sets += list(entry.get("sets", []))
                    sets.append(f"results_dir={results_dir}")
                    out_method = "base" if eval_only else method
                    jobs.append({"block": block, "method": method, "group_size": g, "seed": seed, "eval_only": eval_only,
                                 "sets": sets, "results_dir": results_dir,
                                 "id": f"{block}-{out_method}-g{g}-s{seed}"})
    ids = [j["id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"block {block}: the same job twice")
    return jobs


def run_argv(job: dict, extra_sets=(), local_data=None) -> list[str]:
    argv = ["--method", "vanilla" if job["eval_only"] else job["method"], "--seed", str(job["seed"])]
    for s in list(extra_sets) + job["sets"]:
        argv += ["--set", s]
    if job["eval_only"]:
        argv.append("--eval-only")
    if local_data:
        argv += ["--local-data", *local_data]
    return argv


# ---- what is done -----------------------------------------------------------------------------------------

def expected_config(job: dict, extra_sets=()) -> dict:
    """The config this job would run with now, as run.py resolves it."""
    cfg = load_config("vanilla" if job["eval_only"] else job["method"], list(extra_sets) + job["sets"])
    if job["eval_only"]:
        cfg["method"] = "base"
    return cfg


def matches(run_dir: Path, job: dict, extra_sets=()) -> bool:
    """Is this finished run's resolved config the job's, apart from memory-only settings?"""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return not differences(cfg, expected_config(job, extra_sets))


def job_state(job: dict, root: Path = ROOT, extra_sets=()) -> tuple[str, Path | None]:
    """('done', run) | ('stale', run) | ('todo', None). Stale: finished, but with other values."""
    base = Path(job["results_dir"])
    base = base if base.is_absolute() else root / base
    folder = base / ("base" if job["eval_only"] else job["method"]) / f"g{job['group_size']}"
    finished = sorted(folder.glob(f"seed{job['seed']}-*/summary.json"), key=lambda p: p.parent.name)
    for summary in reversed(finished):                 # newest first
        if matches(summary.parent, job, extra_sets):
            return "done", summary.parent
    return ("stale", finished[-1].parent) if finished else ("todo", None)


# ---- claims (several workers, one grid) -----------------------------------------------------------------

def claims_dir(root: Path) -> Path:
    return root / "results" / ".claims"


TOKEN = uuid.uuid4().hex                       # this worker's identity on its claims


def _age_hours(path: Path) -> float | None:
    try:
        return (time.time() - path.stat().st_mtime) / 3600
    except FileNotFoundError:
        return None


def claim(job: dict, root: Path, stale_hours: float) -> bool:
    """Atomically claim a job. True if this process now owns it.

    A stale claim (a killed job) is first renamed to a name only this worker uses: of several workers
    trying at once, one rename wins and the others find nothing. If what was renamed turns out to be fresh
    (another worker re-claimed in between), it is put back with an atomic link and this worker backs off.
    """
    d = claims_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{job['id']}.json"
    age = _age_hours(path)
    if age is not None and age > stale_hours:
        grave = d / f"{job['id']}.taken-{TOKEN}"
        try:
            os.rename(path, grave)
        except FileNotFoundError:
            return False                               # another worker took it over first
        if (_age_hours(grave) or 0.0) <= stale_hours:  # we grabbed a fresh claim: give it back
            try:
                os.link(grave, path)
            except FileExistsError:
                pass
            grave.unlink(missing_ok=True)
            return False
        grave.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"token": TOKEN, "host": socket.gethostname(), "pid": os.getpid(),
                   "scheduler_job": os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"),
                   "claimed_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, f)
    return True


def release(job: dict, root: Path) -> None:
    """Remove the job's claim, only if it is this worker's."""
    path = claims_dir(root) / f"{job['id']}.json"
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("token") == TOKEN:
            path.unlink(missing_ok=True)
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def is_claimed(job: dict, root: Path, stale_hours: float) -> bool:
    age = _age_hours(claims_dir(root) / f"{job['id']}.json")
    return age is not None and age <= stale_hours


# ---- running ----------------------------------------------------------------------------------------------

def subprocess_runner(argv: list[str]) -> int:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run.py"), *argv], cwd=ROOT).returncode


def check_tree(allow_dirty: bool, tuned_file: Path | None = None) -> None:
    """Refuse to run from uncommitted code, or with a tuned.yaml that is not committed."""
    if allow_dirty:
        return
    state = git_state()
    if state["commit"] is None:
        raise SystemExit("refusing to run: not a git checkout (results must name the commit that made them)")
    if state["dirty"]:
        raise SystemExit("refusing to run: uncommitted changes in tracked files. Commit (and push) first, "
                         "so every result names the code that made it")
    if tuned_file is not None:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(tuned_file)], cwd=ROOT,
                                 capture_output=True, text=True).returncode == 0
        if not tracked:
            raise SystemExit(f"refusing to run: {tuned_file} is not committed. Commit it, so the tuned values "
                             "are part of the code version every result names")


def log_failure(job: dict, root: Path, code: int) -> None:
    d = root / "results" / ".failed"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{job['id']}.txt", "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} exit {code} "
                f"on {socket.gethostname()} job {os.environ.get('PBS_JOBID') or os.environ.get('SLURM_JOB_ID')}\n")


def run_job(job, root, runner, extra_sets, local_data, stale_hours) -> str:
    """'skipped' (done or someone else's), 'ran' or 'failed'."""
    if job_state(job, root, extra_sets)[0] == "done" or not claim(job, root, stale_hours):
        return "skipped"
    try:
        if job_state(job, root, extra_sets)[0] == "done":   # finished between our check and the claim
            return "skipped"
        print(f"\n===== {job['id']}: {' '.join(run_argv(job, extra_sets, local_data))}", flush=True)
        code = runner(run_argv(job, extra_sets, local_data))
        if code != 0 or job_state(job, root, extra_sets)[0] != "done":
            log_failure(job, root, code)
            return "failed"
        return "ran"
    finally:
        release(job, root)


def worker(jobs, root, runner, hours, minutes_per_run, extra_sets=(), local_data=None, stale_hours=12.0) -> dict:
    """Run pending jobs in order until none is left or the next one might not finish in time."""
    start, counts, failed_here = time.time(), {"ran": 0, "failed": 0, "skipped": 0}, set()
    for job in jobs:
        elapsed_h = (time.time() - start) / 3600
        if hours and elapsed_h + minutes_per_run / 60 > hours:
            print(f"stopping: {elapsed_h:.2f} h used of {hours} h, a run takes about {minutes_per_run} min", flush=True)
            break
        if job["id"] in failed_here:
            continue
        result = run_job(job, root, runner, extra_sets, local_data, stale_hours)
        counts[result] += 1
        if result == "failed":
            failed_here.add(job["id"])
    print(f"worker done: {counts}", flush=True)
    return counts


def status(grid, blocks, tuned, root, stale_hours, extra_sets=()) -> dict:
    out = {}
    print("| Block | Jobs | Done | Stale | Running | To do | Failed before |\n|---|---|---|---|---|---|---|")
    for block in blocks:
        try:
            jobs = jobs_for(grid, block, tuned)
        except LookupError as e:
            print(f"| {block} | - | - | - | - | - | needs tuning: {e} |")
            out[block] = {"error": str(e)}
            continue
        states = [job_state(j, root, extra_sets)[0] for j in jobs]
        running = sum(is_claimed(j, root, stale_hours) for j, s in zip(jobs, states) if s != "done")
        failed = sum((root / "results" / ".failed" / f"{j['id']}.txt").exists() for j, s in zip(jobs, states) if s != "done")
        row = {"jobs": len(jobs), "done": states.count("done"), "stale": states.count("stale"), "running": running,
               "todo": len(jobs) - states.count("done") - running, "failed_before": failed}
        out[block] = row
        print(f"| {block} | {row['jobs']} | {row['done']} | {row['stale']} | {row['running']} | {row['todo']} | {failed} |")
    return out


def main(argv=None, runner=None, root: Path = ROOT):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["status", "list", "worker", "run-one"])
    p.add_argument("--block", nargs="*", help="default: every block, in the file's order")
    p.add_argument("--grid", default=str(GRID_FILE))
    p.add_argument("--tuned", default=str(TUNED_FILE))
    p.add_argument("--pending", action="store_true", help="list: only jobs not done")
    p.add_argument("--write", help="list: also write the commands, one per line, to this file")
    p.add_argument("--index", type=int, help="run-one: position in the block's job list (from 0)")
    p.add_argument("--hours", type=float, default=0.0, help="worker: stop before a run could pass this many hours")
    p.add_argument("--minutes-per-run", type=float, default=0.0, help="worker: from docs/compute.md")
    p.add_argument("--stale-hours", type=float, default=12.0, help="a claim older than this is a killed job")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="extra override for every job (a machine setting, not a benchmark choice)")
    p.add_argument("--local-data", nargs=2, metavar=("TRAIN_JSONL", "TEST_JSONL"))
    p.add_argument("--allow-dirty", action="store_true", help="tests and dry runs only")
    args = p.parse_args(argv)

    grid, tuned = read_grid(Path(args.grid)), read_tuned(Path(args.tuned))
    blocks = args.block or list(grid["blocks"])
    runner = runner or subprocess_runner
    if args.command == "status":
        return status(grid, blocks, tuned, root, args.stale_hours, args.set)
    jobs = [j for b in blocks for j in jobs_for(grid, b, tuned)]
    for j in jobs:                                         # a bad method name fails before anything runs
        expected_config(j, args.set)
    if args.command == "list":
        lines = []
        for j in jobs:
            state, _ = job_state(j, root, args.set)
            if args.pending and state == "done":
                continue
            lines.append("python scripts/run.py " + " ".join(run_argv(j, args.set, args.local_data)))
            print(f"{j['id']:40s} {state:5s}  {lines[-1]}")
        if args.write:
            Path(args.write).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lines
    uses_tuned = any(grid["blocks"][b].get("use_tuned") for b in blocks)
    check_tree(args.allow_dirty, Path(args.tuned) if uses_tuned else None)
    if args.command == "run-one":
        if args.index is None or len(blocks) != 1:
            raise SystemExit("run-one needs exactly one --block and an --index")
        if not 0 <= args.index < len(jobs):
            raise SystemExit(f"--index {args.index} is outside 0..{len(jobs) - 1}")
        return run_job(jobs[args.index], root, runner, args.set, args.local_data, args.stale_hours)
    return worker(jobs, root, runner, args.hours, args.minutes_per_run, args.set, args.local_data, args.stale_hours)


if __name__ == "__main__":
    out = main()
    sys.exit(1 if out == "failed" or (isinstance(out, dict) and out.get("failed")) else 0)
