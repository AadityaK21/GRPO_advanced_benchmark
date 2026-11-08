"""Every figure and table of the paper, from the finished runs, with one command.

    python scripts/figures.py                         # results/ -> results/figures/
    python scripts/figures.py --results results_1p5b  # the 1.5B check
    python scripts/figures.py --reps 2000             # quick look (the paper uses 50,000)

The numbers come from scripts/aggregate.py (same runs, same statistics), so a figure never disagrees with
the results table. Each figure is written as PDF (for the paper) and PNG, next to a CSV of exactly the
numbers it draws. A figure whose data is missing (no base run yet, no pass@k) is skipped with a note.

    fig1_accuracy_g<G>    test accuracy per method at the falsification group size: IQM and 95% CI,
                          vanilla's IQM and the base model as reference lines, Holm winners filled
    fig2_group_sizes      IQM accuracy against G, one small panel per method, vanilla in every panel
    fig3_collapse         mean share of uniform groups (2605.21125's collapse rate) and of zero-advantage
                          groups during training, against the accuracy gain over the base model
    fig4_passk            pass@k against k per method, with vanilla and the base model (secondary metric)
    table1_g<G>.md/.tex   per-method numbers at G: IQM, difference to vanilla, p, Holm, pass@k, collapse
    table2_null.md/.tex   the null control
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.stats import bootstrap_ci  # noqa: E402

# Validated with the dataviz skill's palette checker (light surface): blue for methods, orange only for
# the one highlighted series; references in neutral ink. Identity is never colour alone: methods are named
# on the axis or in the panel title, and winners are also filled and starred.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONTROLS = {"vanilla", "base", "vanilla_null"}


def _aggregate():
    spec = importlib.util.spec_from_file_location("aggregate", ROOT / "scripts" / "aggregate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    return plt


def _save(fig, out: Path, name: str, rows: list[dict]) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png", out / f"{name}.csv"]
    fig.savefig(paths[0])
    fig.savefig(paths[1])
    import matplotlib.pyplot as plt
    plt.close(fig)
    with open(paths[2], "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return paths


def _cond(agg: dict, method: str, g: int) -> dict | None:
    return next((c for c in agg["conditions"] if c["method"] == method and c["group_size"] == g), None)


def _base_acc(agg: dict) -> float | None:
    base = [c for c in agg["conditions"] if c["method"] == "base"]
    return base[0]["iqm"] if base else None


def _method_at(agg: dict, method: str, g: int) -> dict | None:
    """aggregate.py's rule: the method at G; a method run at exactly one other G (2-GRPO at 2) there;
    otherwise (runs at several other G but none at this one) not shown at G."""
    here = _cond(agg, method, g)
    if here is not None:
        return here
    others = [c for c in agg["conditions"] if c["method"] == method]
    return others[0] if len(others) == 1 else None


def _order(agg: dict, g: int) -> list[str]:
    """Methods shown at G, best IQM first; controls excluded."""
    names = sorted({c["method"] for c in agg["conditions"]} - CONTROLS)
    shown = {m: _method_at(agg, m, g) for m in names}
    return sorted((m for m in shown if shown[m] is not None), key=lambda m: -shown[m]["iqm"])


# ---- figures ----------------------------------------------------------------------------------------------

def fig1(agg: dict, out: Path, g: int) -> tuple[list[Path], str]:
    van = _cond(agg, "vanilla", g)
    methods = _order(agg, g)
    if not methods or van is None:
        return [], "fig1 skipped: needs vanilla and at least one method at this group size"
    plt = _style()
    base = _base_acc(agg)
    fig, ax = plt.subplots(figsize=(3.4, 0.28 * (len(methods) + 1) + 0.8), layout="constrained")
    rows = []
    names = ["vanilla"] + methods
    for i, m in enumerate(names):
        c = _method_at(agg, m, g)
        t = agg["test"].get(m, {})
        win = bool(t.get("beats_vanilla"))
        y = len(names) - 1 - i
        lo, hi = c["ci"]
        ax.plot([lo, hi], [y, y], color=INK_2 if m == "vanilla" else BLUE, linewidth=1.5, solid_capstyle="round")
        ax.plot(c["iqm"], y, "o", markersize=5, color=INK_2 if m == "vanilla" else BLUE,
                markerfacecolor=(INK_2 if m == "vanilla" else (BLUE if win else "white")), markeredgewidth=1.5)
        rows.append({"method": m, "group_size": c["group_size"], "seeds": len(c["accuracies"]), "iqm": c["iqm"],
                     "ci_low": lo, "ci_high": hi, "beats_vanilla": win})
    ax.axvline(van["iqm"], color=INK_2, linewidth=1, linestyle="--", zorder=0)
    if base is not None:
        ax.axvline(base, color=INK_2, linewidth=1, linestyle=":", zorder=0)
        ax.text(base, len(names) - 0.4, " base", color=INK_2, fontsize=7, va="bottom", ha="left")
    ax.text(van["iqm"], len(names) - 0.4, " vanilla", color=INK_2, fontsize=7, va="bottom", ha="left")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{m}{' *' if agg['test'].get(m, {}).get('beats_vanilla') else ''}"
                        + (f" (G {_method_at(agg, m, g)['group_size']})" if _method_at(agg, m, g)['group_size'] != g else "")
                        for m in reversed(names)])
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"GSM8K test accuracy, {agg['metric']} weights (IQM over seeds, 95% CI)")
    ax.set_title(f"Accuracy per method at G = {g}  (* beats vanilla after Holm)", loc="left")
    return _save(fig, out, f"fig1_accuracy_g{g}", rows), "fig1 written"


def fig2(agg: dict, out: Path) -> tuple[list[Path], str]:
    van = {c["group_size"]: c for c in agg["conditions"] if c["method"] == "vanilla"}
    methods = sorted({c["method"] for c in agg["conditions"]} - CONTROLS)
    methods = [m for m in methods if len({c["group_size"] for c in agg["conditions"] if c["method"] == m}) > 1]
    if not methods or len(van) < 2:
        return [], "fig2 skipped: needs vanilla and a method at two or more group sizes"
    plt = _style()
    cols = min(4, len(methods))
    nrows = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(1.7 * cols + 0.4, 1.45 * nrows + 0.5), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    gs = sorted(van)
    rows = []
    for ax, m in zip(axes.flat, methods):
        conds = sorted((c for c in agg["conditions"] if c["method"] == m), key=lambda c: c["group_size"])
        ax.plot(gs, [van[g]["iqm"] for g in gs], color=INK_2, linestyle="--", linewidth=1)
        ax.fill_between(gs, [van[g]["ci"][0] for g in gs], [van[g]["ci"][1] for g in gs], color=GRID, linewidth=0)
        x = [c["group_size"] for c in conds]
        ax.plot(x, [c["iqm"] for c in conds], color=BLUE, marker="o", markersize=4)
        ax.vlines(x, [c["ci"][0] for c in conds], [c["ci"][1] for c in conds], color=BLUE, linewidth=1.5)
        ax.set_title(m, loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(gs)
        ax.set_xticklabels([str(g) for g in gs])
        ax.minorticks_off()
        for c in conds:
            rows.append({"method": m, "group_size": c["group_size"], "iqm": c["iqm"], "ci_low": c["ci"][0],
                         "ci_high": c["ci"][1], "vanilla_iqm": van.get(c["group_size"], {}).get("iqm")})
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    fig.supxlabel("group size G", fontsize=8)
    fig.supylabel("test accuracy (IQM, 95% CI)", fontsize=8)
    fig.suptitle("Across group sizes (dashed and shaded: vanilla)", x=0.01, ha="left", fontsize=8)
    return _save(fig, out, "fig2_group_sizes", rows), "fig2 written"


def _train_means(run_dir: Path) -> dict:
    """Mean share of uniform groups and of zero-advantage groups over a run's training steps.

    With dynamic sampling (DAPO) the logged per-step shares are over the KEPT groups, which are mixed by
    construction, and a step with no mixed group is logged as skipped. So for those runs the shares are
    counted over every sampled group: dropped groups were all uniform and gave no gradient.
    """
    path = run_dir / "train.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any("groups_dropped" in r for r in rows):        # logged on every step, only with dynamic sampling
        dropped = sum(r.get("groups_dropped", 0) for r in rows)
        kept = sum(r.get("groups_kept", 0) for r in rows)
        if dropped + kept == 0:
            return {}
        uniform_kept = sum(r.get("uniform_frac", 0.0) * r.get("groups_kept", 0) for r in rows if not r.get("skipped"))
        zero_kept = sum(r.get("zero_adv_frac", 0.0) * r.get("groups_kept", 0) for r in rows if not r.get("skipped"))
        return {"uniform_frac": (dropped + uniform_kept) / (dropped + kept),
                "zero_adv_frac": (dropped + zero_kept) / (dropped + kept)}
    rows = [r for r in rows if not r.get("skipped")]
    out = {}
    for key in ("uniform_frac", "zero_adv_frac"):
        vals = [r[key] for r in rows if key in r]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def fig3(agg: dict, chosen: dict, out: Path, g: int, metric: str, reps: int) -> tuple[list[Path], str]:
    base = _base_acc(agg)
    if base is None:
        return [], "fig3 skipped: needs the base model's run (the gain is measured from it)"
    per_method, run_rows = {}, []
    for (m, size), by_seed in chosen.items():
        at = _method_at(agg, m, g)
        if m in ("base", "vanilla_null") or at is None or size != at["group_size"]:
            continue
        for seed, r in sorted(by_seed.items()):
            means = _train_means(r["dir"])
            if "uniform_frac" not in means:
                continue
            gain = r["summary"][metric]["test_accuracy"] - base
            run_rows.append({"method": m, "group_size": size, "seed": seed, "gain": gain, **means})
            per_method.setdefault(m, []).append(run_rows[-1])
    if len(per_method) < 2:
        return [], "fig3 skipped: needs training logs of at least two methods"
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True, layout="constrained")
    rows = []
    for m, rs in sorted(per_method.items()):
        gains = [r["gain"] for r in rs]
        est, lo, hi = bootstrap_ci(gains, reps=reps) if len(gains) > 1 else (gains[0], float("nan"), float("nan"))
        row = {"method": m, "runs": len(rs), "gain_iqm": est, "gain_ci_low": lo, "gain_ci_high": hi}
        for key in ("uniform_frac", "zero_adv_frac"):
            vals = [r[key] for r in rs if key in r]
            row[key] = float(np.mean(vals)) if vals else float("nan")
        rows.append(row)
    notes = []
    for ax, key, label in ((axes[0], "uniform_frac", "share of groups with equal rewards\n(2605.21125's collapse rate)"),
                           (axes[1], "zero_adv_frac", "share of groups with zero advantage\n(depends on the method)")):
        xs = [r[key] for r in run_rows if key in r]
        ys = [r["gain"] for r in run_rows if key in r]
        ax.scatter(xs, ys, s=10, color=GRID, edgecolors=INK_2, linewidths=0.3, zorder=1)
        for row in rows:
            if math.isnan(row[key]):
                continue
            ax.plot([row[key]] * 2, [row["gain_ci_low"], row["gain_ci_high"]], color=BLUE, linewidth=1.5, zorder=2)
            ax.plot(row[key], row["gain_iqm"], "o", color=BLUE, markersize=5, zorder=3)
        _direct_labels(ax, [(row[key], row["gain_iqm"], row["method"]) for row in rows if not math.isnan(row[key])])
        r_runs = _pearson(xs, ys)
        means = [(row[key], row["gain_iqm"]) for row in rows if not math.isnan(row[key])]
        r_methods = _pearson([a for a, _ in means], [b for _, b in means])
        notes.append(f"{key}: r = {r_runs:.2f} over {len(xs)} runs, r = {r_methods:.2f} over {len(means)} methods")
        ax.set_title(f"r = {r_runs:.2f} over runs, {r_methods:.2f} over methods", loc="left")
        ax.set_xlabel(label)
    axes[0].set_ylabel("accuracy gain over base\n(IQM, 95% CI; grey: single runs)")
    for r in rows:
        r["note"] = ""
    rows[0]["note"] = " | ".join(notes)
    return _save(fig, out, "fig3_collapse", rows), "fig3 written: " + "; ".join(notes)


def _direct_labels(ax, points) -> None:
    """Name each point beside it, nudging a label up until it overlaps no earlier one (16 methods crowd)."""
    fig = ax.figure
    fig.canvas.draw()                                  # settle limits and layout before measuring
    pt = fig.dpi / 72
    height, placed = 7.5 * pt, []
    for x, y, text in sorted(points, key=lambda p: p[1]):
        px, py = ax.transData.transform((x, y))
        x0, width, dy = px + 4 * pt, len(text) * 3.3 * pt, 1 * pt
        while any(x0 < c and x0 + width > a and py + dy < d and py + dy + height > b for a, b, c, d in placed):
            dy += height
        placed.append((x0, py + dy, x0 + width, py + dy + height))
        ax.annotate(text, (x, y), xytext=(4, dy / pt), textcoords="offset points", fontsize=6, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.4, shrinkA=0, shrinkB=3) if dy > 2 * pt else None)


def _pearson(x, y) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fig4(agg: dict, out: Path, g: int) -> tuple[list[Path], str]:
    van = _cond(agg, "vanilla", g)
    if van is None or not van["passk"]:
        return [], "fig4 skipped: needs pass@k for vanilla (eval.passk_samples > 0)"
    base = next((c for c in agg["conditions"] if c["method"] == "base" and c["passk"]), None)
    methods = [m for m in _order(agg, g) if _method_at(agg, m, g)["passk"]]
    if not methods:
        return [], "fig4 skipped: no method has pass@k"
    plt = _style()
    cols = min(4, len(methods))
    nrows = math.ceil(len(methods) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(1.7 * cols + 0.4, 1.45 * nrows + 0.5), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")

    def curve(c):
        ks = sorted(c["passk"], key=lambda k: int(k.split("@")[1]))
        return [int(k.split("@")[1]) for k in ks], [c["passk"][k]["iqm"] for k in ks], ks

    rows = []
    vk, vy, _ = curve(van)
    for ax, m in zip(axes.flat, methods):
        c = _method_at(agg, m, g)
        k, y, keys = curve(c)
        ax.plot(vk, vy, color=INK_2, linestyle="--", linewidth=1)
        if base is not None:
            bk, by, _ = curve(base)
            ax.plot(bk, by, color=INK_2, linestyle=":", linewidth=1)
        ax.plot(k, y, color=BLUE, marker="o", markersize=4)
        ax.vlines(k, [c["passk"][kk]["ci"][0] for kk in keys], [c["passk"][kk]["ci"][1] for kk in keys], color=BLUE)
        ax.set_xscale("log", base=2)
        ax.set_xticks(k)
        ax.set_xticklabels([str(x) for x in k])
        ax.minorticks_off()
        ax.set_title(m, loc="left")
        for kk, yy in zip(keys, y):
            rows.append({"method": m, "k": kk, "iqm": yy, "ci_low": c["passk"][kk]["ci"][0],
                         "ci_high": c["passk"][kk]["ci"][1], "vanilla": van["passk"].get(kk, {}).get("iqm"),
                         "base": (base["passk"].get(kk, {}).get("iqm") if base else None)})
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    fig.supxlabel("k (samples per problem)", fontsize=8)
    fig.supylabel("pass@k (IQM, 95% CI)", fontsize=8)
    fig.suptitle(f"pass@k at G = {g} (dashed: vanilla, dotted: base)", x=0.01, ha="left", fontsize=8)
    return _save(fig, out, "fig4_passk", rows), "fig4 written"


# ---- tables -----------------------------------------------------------------------------------------------

def tables(agg: dict, chosen: dict, out: Path, g: int) -> tuple[list[Path], list[str]]:
    out.mkdir(parents=True, exist_ok=True)
    written, notes = [], []
    van = _cond(agg, "vanilla", g)
    if van is not None:
        names = ["vanilla"] + _order(agg, g)
        ks = sorted({k for n in names for k in (_method_at(agg, n, g) or {}).get("passk", {})},
                    key=lambda k: int(k.split("@")[1]))
        ks = [k for k in ks if k in ("pass@1", ks[-1])] if ks else []
        head = ["Method", "G", "Seeds", "IQM [95% CI]", "vs vanilla [95% CI]", "p", "Holm"] + ks + ["Collapse"]
        md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        tex = [r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}", r"\toprule",
               " & ".join(h.replace("%", r"\%") for h in head) + r" \\", r"\midrule"]
        for n in names:
            c = _method_at(agg, n, g)
            t = agg["test"].get(n)
            collapse = [m.get("uniform_frac") for r in chosen.get((n, c["group_size"]), {}).values()
                        for m in [_train_means(r["dir"])] if "uniform_frac" in m]
            cells = [n, str(c["group_size"]), str(len(c["accuracies"])),
                     f"{c['iqm']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]",
                     f"{t['diff']:+.3f} [{t['low']:+.3f}, {t['high']:+.3f}]" if t else "-",
                     f"{t['p_value']:.3f}" if t else "-",
                     ("yes" if t["beats_vanilla"] else "no") if t else "-"]
            cells += [f"{c['passk'][k]['iqm']:.3f}" if k in c.get("passk", {}) else "-" for k in ks]
            cells.append(f"{np.mean(collapse):.2f}" if collapse else "-")
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(x.replace("_", r"\_") for x in cells) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}"]
        for name, text in ((f"table1_g{g}.md", "\n".join(md)), (f"table1_g{g}.tex", "\n".join(tex))):
            (out / name).write_text(text + "\n", encoding="utf-8")
            written.append(out / name)
    else:
        notes.append("table1 skipped: no vanilla runs at this group size")
    nc = agg.get("null_control")
    if nc:
        md = ["| Runs pooled (vanilla + null) | Random splits excluding 0 | Expected |", "|---|---|---|",
              f"| {nc['runs']} | {nc['exclusion_rate']:.1%} | about 5% |"]
        tex = [r"\begin{tabular}{rrr}", r"\toprule", r"Runs pooled & Splits excluding 0 & Expected \\", r"\midrule",
               f"{nc['runs']} & {100 * nc['exclusion_rate']:.1f}\\% & 5\\% \\\\", r"\bottomrule", r"\end{tabular}"]
        for name, text in (("table2_null.md", "\n".join(md)), ("table2_null.tex", "\n".join(tex))):
            (out / name).write_text(text + "\n", encoding="utf-8")
            written.append(out / name)
    else:
        notes.append("table2 skipped: no null-control runs (vanilla_null) at this group size")
    return written, notes


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--out", default=None, help="default: <results>/figures")
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--metric", choices=["final", "best"], default="final")
    p.add_argument("--reps", type=int, default=50_000)
    args = p.parse_args(argv)
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "figures"
    aggregate = _aggregate()
    agg = aggregate.main(["--results", str(results), "--metric", args.metric, "--group", str(args.group),
                          "--reps", str(args.reps), "--no-write"])     # never overwrite the results table
    chosen, _ = aggregate.collect(results)
    if not agg["conditions"]:
        print(f"no finished runs under {results}: nothing to draw")
        return {"written": [], "notes": ["no runs"]}
    written, notes = [], []
    for paths, note in (fig1(agg, out, args.group), fig2(agg, out),
                        fig3(agg, chosen, out, args.group, args.metric, args.reps), fig4(agg, out, args.group)):
        written += paths
        notes.append(note)
    t_paths, t_notes = tables(agg, chosen, out, args.group)
    written += t_paths
    notes += t_notes
    print("\n".join(notes))
    print(f"{len(written)} files in {out}")
    return {"written": [str(p) for p in written], "notes": notes}


if __name__ == "__main__":
    main()
