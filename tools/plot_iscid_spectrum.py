#!/usr/bin/env python3
"""ISCID Figure 2: 2x2 history-budget spectrum (frozen-JSON driven).

Top row (a) New York, (b) Chicago: generalist-minus-ranker Recall@20%
margins over the nine history budgets under the frozen citywide convention
(per-partition standardised frequency) and within-partition stratified
allocation, plus the raw-frequency citywide ranker as a convention
sensitivity (dashed).
Bottom row (c) New York, (d) Chicago: absolute Recall@20% of the ranker and
the generalist under stratified allocation.

Sources (read-only, frozen):
  - analysis_results/formal_main_table.json .headline_spectrum (NYC stratified)
  - analysis_results/cross_city_spectrum_nyc_citywide.json      (NYC citywide)
  - analysis_results/cross_city_spectrum_chicago_citywide.json  (Chicago citywide)
  - analysis_results/cross_city_spectrum_chicago_stratified.json (Chicago stratified)
  - analysis_results/citywide_ranker_variants_nyc.json      (raw-frequency ranker)
  - analysis_results/citywide_ranker_variants_chicago.json  (raw-frequency ranker)
Output: analysis_results/paper_figures/figure_iscid_spectrum.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AR = ROOT / "analysis_results"
OUT = AR / "paper_figures"
BUDGETS = ["14", "21", "30", "60", "90", "120", "180", "365", "0"]
LABELS = ["14", "21", "30", "60", "90", "120", "180", "365", "full"]

STRAT = ("o-", "#0072B2", "stratified")
RAW = ("s-", "#E69F00", "citywide (raw freq., primary)")
CITY = ("^--", "#999999", "citywide (std.)")
RANKER = ("s-", "#009E73", "frequency baseline")
GENERALIST = ("o-", "#D55E00", "mixed-budget model")


def nyc_stratified_margin() -> dict[str, float]:
    rows = json.loads((AR / "formal_main_table.json").read_text())["headline_spectrum"]
    by_l = {str(r["L"]): r for r in rows}
    return {b: by_l[b]["margin_R20"] for b in BUDGETS if b in by_l}


def nyc_stratified_absolute() -> tuple[dict[str, float], dict[str, float]]:
    rows = json.loads((AR / "formal_main_table.json").read_text())["headline_spectrum"]
    by_l = {str(r["L"]): r for r in rows}
    ranker = {b: by_l[b]["ranker_R20"] for b in BUDGETS if b in by_l}
    generalist = {b: by_l[b]["generalist_R20"] for b in BUDGETS if b in by_l}
    return ranker, generalist


def cross_city_margin(name: str) -> dict[str, float]:
    doc = json.loads((AR / f"cross_city_spectrum_{name}.json").read_text())
    out = {}
    for b in BUDGETS:
        cell = doc["by_budget"].get(b)
        if cell is None:
            continue  # Chicago stratified was frozen at five budgets only
        out[b] = cell["generalist"]["es_recall_at_20"] - cell["ranker"]["es_recall_at_20"]
    return out


def cross_city_absolute(name: str) -> tuple[dict[str, float], dict[str, float]]:
    doc = json.loads((AR / f"cross_city_spectrum_{name}.json").read_text())
    ranker, generalist = {}, {}
    for b in BUDGETS:
        cell = doc["by_budget"].get(b)
        if cell is None:
            continue
        ranker[b] = cell["ranker"]["es_recall_at_20"]
        generalist[b] = cell["generalist"]["es_recall_at_20"]
    return ranker, generalist


def raw_city_margin(name: str) -> dict[str, float]:
    gen_doc = json.loads((AR / f"cross_city_spectrum_{name}_citywide.json").read_text())
    raw_doc = json.loads((AR / f"citywide_ranker_variants_{name}.json").read_text())
    out = {}
    for b in BUDGETS:
        cell = gen_doc["by_budget"].get(b)
        raw_cell = raw_doc["by_budget"].get(b)
        if cell is None or raw_cell is None:
            continue
        out[b] = cell["generalist"]["es_recall_at_20"] - raw_cell["raw_citywide"]["es_recall_at_20"]
    return out


def dodge_offsets(points: list[tuple[float, float]], min_gap: float = 11.0) -> list[tuple[float, float]]:
    """Vertical dodge for endpoint labels given (data_y, display_y) pairs.

    Series that converge at the full-budget end (three conventions within a few
    display points) get label anchors pushed apart to ``min_gap`` display
    points, keeping their relative order.
    """
    order = sorted(range(len(points)), key=lambda i: points[i][1])
    target = [points[i][1] for i in order]
    for i in range(1, len(target)):
        if target[i] - target[i - 1] < min_gap:
            target[i] = target[i - 1] + min_gap
    # recentre the pushed-apart cluster on its original centroid
    shift = sum(points[i][1] for i in order) / len(order) - sum(target) / len(target)
    out = [None] * len(points)
    for slot, idx in enumerate(order):
        out[idx] = (5.0, target[slot] + shift - points[idx][1])
    return out


def margin_panel(ax, series: dict, title: str, dodge: bool = False) -> None:
    fmt_map = {"strat": STRAT, "city": CITY, "raw": RAW}
    keys = list(series)
    ends = []
    for key in keys:
        fmt, color, label = fmt_map[key]
        by_budget = series[key]
        xs = [BUDGETS.index(b) for b in by_budget]
        ys = list(by_budget.values())
        ax.plot(xs, ys, fmt, color=color, markersize=3.0, linewidth=1.0,
                label=label, zorder=3)
        ends.append((xs[-1], ys[-1]))
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    # fix the y-range before reading display coordinates so the endpoint-label
    # dodge is computed under the final transform
    ax.set_ylim(-0.05, 0.38)
    if dodge and len(ends) >= 2:
        disp = [ax.transData.transform((x, y))[1] for x, y in ends]
        offsets = dodge_offsets([(y, d) for (x, y), d in zip(ends, disp)])
    else:
        offsets = [(5, 4 if k == "strat" else -8) for k in keys]
    for key, (dx, dy), (x_end, y_end) in zip(keys, offsets, ends):
        color = fmt_map[key][1]
        ax.annotate(f"{y_end:+.3f}", (x_end, y_end), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.0, color=color)
    ax.set_xticks(list(range(len(BUDGETS))))
    ax.set_xticklabels(LABELS, fontsize=6.4)
    ax.tick_params(axis="y", labelsize=6.4)
    ax.set_xlabel("History budget $L$ (days)", fontsize=6.8)
    ax.set_title(title, fontsize=7.0, pad=3)
    ax.grid(alpha=0.3, linewidth=0.5, zorder=0)


def absolute_panel(ax, series: dict, title: str) -> None:
    keys = list(series)
    ends = []
    for key in keys:
        fmt, color, label = {"ranker": RANKER, "generalist": GENERALIST}[key]
        by_budget = series[key]
        xs = [BUDGETS.index(b) for b in by_budget]
        ys = list(by_budget.values())
        ax.plot(xs, ys, fmt, color=color, markersize=3.0, linewidth=1.0,
                label=label, zorder=3)
        ends.append((xs[-1], ys[-1]))
    ax.set_ylim(0.2, 0.75)
    disp = [ax.transData.transform((x, y))[1] for x, y in ends]
    offsets = dodge_offsets([(y, d) for (x, y), d in zip(ends, disp)])
    for key, (dx, dy), (x_end, y_end) in zip(keys, offsets, ends):
        color = {"ranker": RANKER, "generalist": GENERALIST}[key][1]
        ax.annotate(f"{y_end:.3f}", (x_end, y_end), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.0, color=color)
    ax.set_xticks(list(range(len(BUDGETS))))
    ax.set_xticklabels(LABELS, fontsize=6.4)
    ax.tick_params(axis="y", labelsize=6.4)
    ax.set_xlabel("History budget $L$ (days)", fontsize=6.8)
    ax.set_title(title, fontsize=7.0, pad=3)
    ax.grid(alpha=0.3, linewidth=0.5, zorder=0)


# --- single-column margins-only layout (conference main text) ---
# Native \columnwidth rendering needs >=6.5pt type (legibility),
# so this variant is drawn at final size instead of being scaled down.
STRAT1 = ("o-", "#0072B2", "stratified")
RAW1 = ("s-", "#E69F00", "citywide (raw)")
CITY1 = ("^--", "#999999", "citywide (std.)")


def margin_panel_1col(ax, series: dict, title: str, fs: float = 6.5) -> None:
    fmt_map = {"strat": STRAT1, "city": CITY1, "raw": RAW1}
    keys = list(series)
    ends = []
    for key in keys:
        fmt, color, label = fmt_map[key]
        by_budget = series[key]
        xs = [BUDGETS.index(b) for b in by_budget]
        ys = list(by_budget.values())
        ax.plot(xs, ys, fmt, color=color, markersize=2.4, linewidth=1.0,
                label=label, zorder=3)
        ends.append((xs[-1], ys[-1]))
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.set_ylim(-0.09, 0.42)
    ax.set_xlim(-0.35, 9.6)  # endpoint labels sit centred on the last point
    # endpoint values stacked above the flat tail (rank-ordered) so the text
    # never covers the curves or the zero line
    order = sorted(range(len(keys)), key=lambda i: ends[i][1])
    rank = {idx: r for r, idx in enumerate(order)}
    for pos, (key, (x_end, y_end)) in enumerate(zip(keys, ends)):
        color = fmt_map[key][1]
        dy = 10 + 7 * rank[pos]
        ax.annotate(f"{y_end:+.3f}", (x_end, y_end), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=fs - 0.5, color=color,
                    zorder=4)
    ax.legend(loc="upper right", fontsize=fs - 0.5, frameon=False,
              handlelength=1.4, handletextpad=0.4, labelspacing=0.25,
              borderpad=0.1)
    tick_idx = [0, 2, 4, 6, 8]  # sparse ticks: nine labels collide at column width
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([LABELS[i] for i in tick_idx], fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xlabel("History budget $L$ (days)", fontsize=fs)
    ax.set_title(title, fontsize=fs + 0.5, pad=2)
    ax.grid(alpha=0.3, linewidth=0.5, zorder=0)


def main_margins1col() -> None:
    series_a = {
        "strat": nyc_stratified_margin(),
        "city": cross_city_margin("nyc_citywide"),
        "raw": raw_city_margin("nyc"),
    }
    series_b = {
        "city": cross_city_margin("chicago_citywide"),
        "strat": cross_city_margin("chicago_stratified"),
        "raw": raw_city_margin("chicago"),
    }
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.45, 1.55),
                                     sharex=True, sharey=True)
    margin_panel_1col(ax_a, series_a, "(a) New York")
    margin_panel_1col(ax_b, series_b, "(b) Chicago")
    ax_a.set_ylabel("Recall@20% margin", fontsize=6.5)
    fig.tight_layout(pad=0.35)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure_iscid_spectrum_margins.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"saved {OUT}/figure_iscid_spectrum_margins.{{pdf,png}}")


def main() -> None:
    nyc_ranker, nyc_gen = nyc_stratified_absolute()
    chi_ranker, chi_gen = cross_city_absolute("chicago_stratified")
    series = {
        "a": {
            "strat": nyc_stratified_margin(),
            "city": cross_city_margin("nyc_citywide"),
            "raw": raw_city_margin("nyc"),
        },
        "b": {
            "city": cross_city_margin("chicago_citywide"),
            "strat": cross_city_margin("chicago_stratified"),
            "raw": raw_city_margin("chicago"),
        },
        "c": {"ranker": nyc_ranker, "generalist": nyc_gen},
        "d": {"ranker": chi_ranker, "generalist": chi_gen},
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3), sharex=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    margin_panel(ax_a, series["a"], "(a) New York: margin over baseline", dodge=True)
    margin_panel(ax_b, series["b"], "(b) Chicago: margin over baseline", dodge=True)
    absolute_panel(ax_c, series["c"], "(c) New York: absolute Recall@20%")
    absolute_panel(ax_d, series["d"], "(d) Chicago: absolute Recall@20%")
    ax_a.set_ylabel("Margin over baseline\nRecall@20%", fontsize=6.8)
    ax_b.set_ylabel("")
    ax_c.set_ylabel("Recall@20%", fontsize=6.8)
    ax_d.set_ylabel("")
    handles, labels = ax_a.get_legend_handles_labels()
    handles2, labels2 = ax_c.get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="lower center", ncol=5,
               fontsize=6.4, framealpha=0.9, handlelength=1.6,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.5, rect=(0, 0.05, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure_iscid_spectrum.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {OUT}/figure_iscid_spectrum.{{pdf,png}}")
    for panel, group in series.items():
        for key, by_budget in group.items():
            print(f"{panel}/{key:8s}: {[round(y, 4) for y in by_budget.values()]}")


if __name__ == "__main__":
    import sys

    layout = sys.argv[1] if len(sys.argv) > 1 else "2x2"
    if layout == "margins1col":
        main_margins1col()
    else:
        main()
