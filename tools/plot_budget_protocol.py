#!/usr/bin/env python3
"""Protocol schematic: the budget-matched history-budget comparison (no data input).

Draws the evaluation protocol of main-text Sections 2.1-2.2 as a flow diagram:
left-truncated crash archive at forecast origin t -> two scorers reading exactly
the same truncated archive -> shared top-k screening -> retrieval metrics.
Pure vector diagram; style follows the other paper figures (small type,
grayscale-safe accents). Output: analysis_results/paper_figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_results/paper_figures"

INK = "#1a1a1a"
GRAY = "#666666"
BLUE = "#0072B2"
LIGHT = "#f2f2f2"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=INK, fs=7.0, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=0.8, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3, weight=weight, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=GRAY):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
        linewidth=0.9, color=color, zorder=1,
        shrinkA=0, shrinkB=0))


def main() -> None:
    # Same horizontal composition as the original double-column schematic
    # (timeline -> two parallel scorers -> shared top-k -> metrics), redrawn at
    # column width: box text is condensed so 6.2pt type stays readable, and
    # both scorer boxes share one neutral style (blue accent reserved for the
    # available-history bracket).
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 104)
    ax.axis("off")

    # --- timeline ---------------------------------------------------------
    ax.plot([4, 55], [90, 90], color=INK, linewidth=1.1, zorder=2)
    ax.plot([55, 55], [87.5, 92.5], color=INK, linewidth=1.1, zorder=2)
    ax.plot([55, 78], [90, 90], color=GRAY, linewidth=1.0, linestyle=(0, (3, 2)), zorder=2)
    ax.text(4, 93.5, "crash archive $\\rightarrow$", fontsize=6.2,
            color=INK, ha="left", va="bottom")
    ax.text(55, 93.5, "origin $t$", fontsize=6.2, color=INK,
            ha="center", va="bottom", weight="bold")
    ax.text(96, 93.5, "future $H{=}3$ days\n(occurrence labels)", fontsize=5.6,
            color=GRAY, ha="right", va="bottom")
    # available-history bracket under the archive line
    ax.plot([33, 33, 55, 55], [86.0, 84.5, 84.5, 86.0], color=BLUE,
            linewidth=0.9, zorder=2)
    ax.text(44, 82.5, "available history: $n_{i,t}=\\min(L_i,\\, t-t_0)$ days "
            "(budget $L$)",
            fontsize=5.8, color=BLUE, ha="center", va="top")

    # --- two scorers (identical styling) -----------------------------------
    arrow(ax, 44, 76.0, 27, 72.5)
    arrow(ax, 44, 76.0, 73, 72.5)
    box(ax, 4, 52, 44, 20,
        "Budget-matched\nfrequency baseline\ntrailing mean, zero training", fs=5.8)
    box(ax, 52, 52, 44, 20,
        "Mixed-budget model\n$T{=}14$ history window +\nstatic/context inputs",
        fs=5.8)

    # --- shared screening -------------------------------------------------
    arrow(ax, 26, 52, 42, 41.5)
    arrow(ax, 74, 52, 58, 41.5)
    box(ax, 16, 27, 68, 14,
        "scores at origin $t$ $\\rightarrow$ top-$k$ share $a\\%$\n"
        "(stratified or citywide allocation)", fs=5.8)
    arrow(ax, 50, 27, 50, 18.5)
    box(ax, 16, 4, 68, 14,
        "Recall@$a$%, NDCG@$a$% vs occurrence labels\n"
        "$\\Rightarrow$ margin (mixed-budget model $-$ baseline)", fs=5.8)

    fig.tight_layout(pad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"figure_budget_protocol.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {OUT}/figure_budget_protocol.{{pdf,png}}")


if __name__ == "__main__":
    main()
