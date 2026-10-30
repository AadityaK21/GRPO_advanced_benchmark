"""Aggregate finished runs: IQM test accuracy with 95% bootstrap intervals, and the falsification test.

    python scripts/aggregate.py                    # everything under results/
    python scripts/aggregate.py --metric best      # the validation-chosen checkpoint instead of the final one

Reads every results/<method>/g<G>/seed<k>-<time>/summary.json. If a seed was run more than once,
the newest finished run counts and the others are listed. Warns when runs that are compared differ in
GPU, dtype or code version, or were run from a tree with uncommitted changes.

The falsification test (README): each method against vanilla at group size --group (default 8), paired
by seed, IQM difference, percentile bootstrap, Holm's correction over the methods in --family.
A method with no runs at that group size but runs at exactly one other (2-GRPO at 2) is compared at that.
Writes results/aggregate.md and results/aggregate.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.stats import bootstrap_ci, holm, null_control, paired_bootstrap  # noqa: E402

CONTROLS = {"vanilla", "base", "vanilla_null"}


def collect(results: Path) -> tuple[dict, list[str]]:
    """{(method, G): {seed: run}} using the newest finished run per seed, and a list of warnings."""
    runs, notes = defaultdict(lambda: defaultdict(list)), []
    for summary in sorted(results.glob("*/g*/seed*/summary.json")):
        run_dir = summary.parent
        s = json.loads(summary.read_text())
        if s.get("val_only"):        # no test numbers: a prompt check or tuning run in the wrong folder
            notes.append(f"{run_dir}: a --val-only run, skipped")
            continue
        meta_file = run_dir / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        runs[(s["method"], s["group_size"])][s["seed"]].append({"dir": run_dir, "summary": s, "meta": meta})
    chosen = {}
    for key, by_seed in runs.items():
        chosen[key] = {}
        for seed, rs in by_seed.items():
            rs.sort(key=lambda r: r["dir"].name)            # the folder name ends with the UTC start time
            chosen[key][seed] = rs[-1]
            if len(rs) > 1:
                notes.append(f"{key[0]} g{key[1]} seed {seed}: {len(rs)} runs, using {rs[-1]['dir'].name}; "
                             f"ignored {', '.join(r['dir'].name for r in rs[:-1])}")
            if rs[-1]["meta"].get("dirty"):
                notes.append(f"{rs[-1]['dir']}: run from a tree with uncommitted changes")
    return chosen, notes


def accuracy(run: dict, metric: str) -> float:
    return run["summary"][metric]["test_accuracy"]


def consistency(runs: list[dict]) -> list[str]:
    notes = []
    for field in ("gpu", "dtype", "commit"):
        values = {r["meta"].get(field) for r in runs}
        if len(values) > 1:
            notes.append(f"runs differ in {field}: {sorted(map(str, values))}")
    return notes


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
    p.add_argument("--metric", choices=["final", "best"], default="final")
    p.add_argument("--group", type=int, default=8, help="group size of the falsification test")
    p.add_argument("--family", nargs="*", default=None,
                   help="methods in the Holm family (default: every method except the controls)")
    p.add_argument("--reps", type=int, default=50_000)
    args = p.parse_args(argv)
    results = Path(args.results)
    chosen, notes = collect(results)
    out = {"metric": args.metric, "conditions": [], "test": {}, "notes": notes}
    lines = [f"# Results ({args.metric} checkpoint, test exact-match)", "",
             "| Method | G | Seeds | IQM accuracy | 95% CI | Rollouts | Minutes / run |", "|---|---|---|---|---|---|---|"]
    for (method, g), by_seed in sorted(chosen.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        accs = [accuracy(r, args.metric) for _, r in sorted(by_seed.items())]
        est, lo, hi = bootstrap_ci(accs, reps=args.reps) if len(accs) > 1 else (accs[0], float("nan"), float("nan"))
        rollouts = sum(r["summary"]["rollouts"] for r in by_seed.values()) / len(by_seed)
        minutes = sum(r["summary"]["wall_clock_seconds"] for r in by_seed.values()) / len(by_seed) / 60
        out["conditions"].append({"method": method, "group_size": g, "seeds": sorted(by_seed), "iqm": est,
                                  "ci": [lo, hi], "accuracies": accs, "mean_rollouts": rollouts})
        lines.append(f"| {method} | {g} | {len(accs)} | {est:.4f} | [{lo:.4f}, {hi:.4f}] | {rollouts:.0f} | {minutes:.1f} |")
        out["notes"] += [f"{method} g{g}: {n}" for n in consistency(list(by_seed.values()))]

    # The falsification test: each method vs vanilla at --group, paired by seed.
    vanilla = chosen.get(("vanilla", args.group), {})
    methods = sorted({m for m, _ in chosen} - CONTROLS) if args.family is None else args.family
    p_values, tests = {}, {}
    for method in methods:
        sizes = sorted(g for m, g in chosen if m == method)
        g = args.group if args.group in sizes else (sizes[0] if len(sizes) == 1 else None)
        if g is None or not vanilla:
            out["notes"].append(f"{method}: no runs to compare with vanilla at g{args.group}")
            continue
        seeds = sorted(set(chosen[(method, g)]) & set(vanilla))
        if len(seeds) < 2:
            out["notes"].append(f"{method}: fewer than 2 seeds shared with vanilla")
            continue
        a = [accuracy(chosen[(method, g)][s], args.metric) for s in seeds]
        b = [accuracy(vanilla[s], args.metric) for s in seeds]
        tests[method] = {**paired_bootstrap(a, b, reps=args.reps), "group_size": g, "seeds": seeds}
        p_values[method] = tests[method]["p_value"]
    rejected = holm(p_values)
    if tests:
        lines += ["", f"## Against vanilla at G={args.group}: paired bootstrap, Holm over {len(tests)} methods", "",
                  "| Method | G | Seeds | IQM difference | 95% CI | p | Beats vanilla |", "|---|---|---|---|---|---|---|"]
        for method, t in tests.items():
            t["beats_vanilla"] = bool(rejected[method] and t["diff"] > 0)
            lines.append(f"| {method} | {t['group_size']} | {len(t['seeds'])} | {t['diff']:+.4f} | "
                         f"[{t['low']:+.4f}, {t['high']:+.4f}] | {t['p_value']:.4f} | {'yes' if t['beats_vanilla'] else 'no'} |")
    out["test"] = tests

    null = chosen.get(("vanilla_null", args.group))
    if null and vanilla:
        pooled = [accuracy(r, args.metric) for r in list(vanilla.values()) + list(null.values())]
        rate = null_control(pooled, splits=1000, reps=2000)
        out["null_control"] = {"runs": len(pooled), "exclusion_rate": rate}
        lines += ["", f"Null control: {rate:.1%} of random vanilla-vs-vanilla splits exclude zero (expect about 5%)."]

    if out["notes"]:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in out["notes"]]
    text = "\n".join(lines) + "\n"
    print(text)
    if results.exists():
        (results / "aggregate.md").write_text(text, encoding="utf-8")
        (results / "aggregate.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
