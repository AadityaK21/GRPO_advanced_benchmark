"""One folder per run, with everything needed to trust or redo it.

results/<method>/g<group size>/seed<seed>-<UTC time>/
    config.yaml     the full resolved config, overrides included
    meta.json       seed, git commit (and whether the tree had uncommitted changes), command,
                    versions, GPU, dtype, start time
    train.jsonl     one line per step: reward, diagnostics, rollouts so far, time, memory
    eval.jsonl      one line per validation check
    summary.json    written last: final and best accuracy, rollouts used, wall-clock.
                    A folder without it is an unfinished run and is ignored by the aggregation
    ckpt/           adapters and completions; large, so git ignores it
A re-run gets a new folder; nothing is overwritten.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

from src.config import ROOT, dump


def git_state(repo: Path = ROOT) -> dict:
    """{'commit': sha or None, 'dirty': bool or None}. dirty means files differ from the commit."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=30)
    try:
        head = run("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": None, "dirty": None}
        status = run("status", "--porcelain", "--untracked-files=no")
        return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def new_run_dir(cfg: dict, seed: int, root: Path | str | None = None) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = Path(root or ROOT / cfg["results_dir"]) / cfg["method"] / f"g{cfg['budget']['group_size']}"
    path = base / f"seed{seed}-{stamp}"
    n = 1
    while path.exists():                     # two runs started in the same second
        n += 1
        path = base / f"seed{seed}-{stamp}-{n}"
    path.mkdir(parents=True)
    return path


def write_start(run_dir: Path, cfg: dict, seed: int, extra: dict | None = None) -> dict:
    import torch
    import transformers

    meta = {
        "seed": seed,
        "method": cfg["method"],
        **git_state(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **(extra or {}),
    }
    try:
        import peft
        meta["peft"] = peft.__version__
    except ImportError:
        pass
    (run_dir / "config.yaml").write_text(dump(cfg), encoding="utf-8")
    write_json(run_dir / "meta.json", meta)
    return meta


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)                        # never a half-written file


class JsonlWriter:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()
