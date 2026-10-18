"""Reward per step for your loop against TRL, over seeds (week 7).

    python scripts/compare_runs.py --ours "runs/ours_seed*" --trl "runs/trl_seed*" --out runs/compare.png

Each panel shows a trailing moving average (--window steps, because one step is one question and
its reward is mostly noise), one thin line per seed and a bold line for the mean over seeds, with
the band between the lowest and highest seed. The two methods agree when their bands overlap.
It also prints the mean of each run over its first and last --window steps.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

COLORS = {"ours": "#2a78d6", "trl": "#eb6834"}   # categorical slots 1 and 2, checked for colour blindness
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load(pattern: str) -> dict[str, dict[str, np.ndarray]]:
    runs = {}
    for d in sorted(glob.glob(pattern)):
        f = Path(d) / "log.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows.sort(key=lambda r: r["step"])
        runs[Path(d).name] = {k: np.array([r.get(k) if r.get(k) is not None else np.nan for r in rows], float)
                              for k in ("step", "reward", "completion_length", "truncated_frac")}
    return runs


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        out[i] = np.nanmean(x[max(0, i - w + 1): i + 1])
    return out


def summary(name, runs, w):
    for run, d in runs.items():
        r = d["reward"]
        print(f"  {name:4s} {run:20s} steps {len(r):4d}   first {w}: {np.nanmean(r[:w]):.3f}   last {w}: {np.nanmean(r[-w:]):.3f}"
              f"   cut off: {np.nanmean(d['truncated_frac']):.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ours", required=True, help='glob of run folders, e.g. "runs/ours_seed*"')
    p.add_argument("--trl", required=True, help='glob of run folders, e.g. "runs/trl_seed*"')
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--out", default="runs/compare.png")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = {"ours": load(args.ours), "trl": load(args.trl)}
    for name, runs in methods.items():
        if not runs:
            raise SystemExit(f"no runs with a log.jsonl match --{name} {getattr(args, name)!r}")
    print(f"mean reward per run (window {args.window}):")
    for name, runs in methods.items():
        summary(name, runs, args.window)

    panels = [("reward", "Reward (exact match), moving average"),
              ("completion_length", "Completion length (tokens), moving average")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    for ax, (key, title) in zip(axes, panels):
        ends = []   # (y at the last step, label), placed after both lines are drawn
        for name, runs in methods.items():
            curves = [trailing_mean(d[key], args.window) for d in runs.values()]
            n = min(len(c) for c in curves)
            stack = np.vstack([c[:n] for c in curves])
            steps = np.arange(n)
            for c in stack:
                ax.plot(steps, c, color=COLORS[name], lw=1, alpha=0.35)
            ax.fill_between(steps, np.nanmin(stack, 0), np.nanmax(stack, 0), color=COLORS[name], alpha=0.12, lw=0)
            mean = np.nanmean(stack, 0)
            label = f"{'Our loop' if name == 'ours' else 'TRL'} ({len(curves)} seeds)"
            ax.plot(steps, mean, color=COLORS[name], lw=2, label=label)
            ends.append([steps[-1], mean[-1], "Our loop" if name == "ours" else "TRL"])
        lo, hi = ax.get_ylim()
        gap = 0.07 * (hi - lo)                    # keep the two end labels from overlapping
        ends.sort(key=lambda e: e[1])
        if len(ends) == 2 and ends[1][1] - ends[0][1] < gap:
            mid = (ends[0][1] + ends[1][1]) / 2
            ends[0][1], ends[1][1] = mid - gap / 2, mid + gap / 2
        for x, y, text in ends:
            ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("step", color=MUTED)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.margins(x=0.08)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
