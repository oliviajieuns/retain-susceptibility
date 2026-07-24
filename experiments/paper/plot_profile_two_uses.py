"""Draw the schematic Figure 2: one pre-unlearning profile, two tests.

This figure is deliberately conceptual.  It contains no empirical effect
sizes, confidence intervals, or pass/fail claims; Tables 1--2 carry those
results.  Figure 1 explains chronology, whereas this figure explains why the
two profile signals are complementary and how the same signal family supports
two distinct estimands.

Usage
-----
python experiments/paper/plot_profile_two_uses.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#707070"
LIGHT_GRAY = "#D9D9D9"
INK = "#1A1A1A"


def _panel_label(ax, letter: str, title: str, subtitle: str) -> None:
    ax.text(
        0.0,
        1.18,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="white",
        fontsize=7.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc=INK, ec="none"),
    )
    ax.text(
        0.075,
        1.18,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.075,
        0.99,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.3,
        color=GRAY,
    )


def _profile_panel(ax) -> None:
    _panel_label(
        ax,
        "A",
        "PROFILE: why both signals are needed",
        r"sensitivity × exposure, frozen at $\theta_0$",
    )

    grid = np.linspace(0, 1, 240)
    qh, qg = np.meshgrid(grid, grid)
    score = 0.48 * qg + 0.52 * qh
    cmap = LinearSegmentedColormap.from_list(
        "risk", ["#FFFFFF", "#F1F6F4", "#D9EEE7", "#B8DFD1"]
    )
    ax.imshow(
        score,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="bilinear",
        zorder=0,
        aspect="auto",
    )
    ax.axhline(0.5, color=LIGHT_GRAY, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.axvline(0.5, color=LIGHT_GRAY, lw=0.8, ls=(0, (3, 3)), zorder=1)

    candidates = np.array(
        [
            (0.14, 0.16),
            (0.27, 0.30),
            (0.39, 0.18),
            (0.19, 0.68),
            (0.36, 0.78),
            (0.58, 0.20),
            (0.62, 0.55),
            (0.83, 0.63),
            (0.88, 0.87),
        ]
    )
    ax.scatter(
        candidates[:, 0],
        candidates[:, 1],
        s=30,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=0.8,
        zorder=3,
    )

    # Same exposure, different susceptibility: the reason proximity alone is
    # insufficient.
    same_x = 0.72
    ax.plot(
        [same_x, same_x],
        [0.27, 0.80],
        color=GRAY,
        lw=0.9,
        ls=(0, (2, 2)),
        zorder=2,
    )
    ax.scatter([same_x], [0.27], s=62, fc="white", ec=BLUE, lw=1.8, zorder=5)
    ax.scatter([same_x], [0.80], s=62, fc=BLUE, ec=BLUE, lw=1.3, zorder=5)
    ax.text(same_x + 0.035, 0.25, "A: locally stable", fontsize=7.2, color=BLUE)
    ax.text(same_x + 0.035, 0.79, "B: locally fragile", fontsize=7.2, color=BLUE)
    ax.text(
        same_x - 0.02,
        0.53,
        "same proximity",
        rotation=90,
        ha="right",
        va="center",
        fontsize=6.7,
        color=GRAY,
    )

    ax.annotate(
        "",
        xy=(0.91, 0.91),
        xytext=(0.12, 0.10),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.7),
        zorder=4,
    )
    ax.text(
        0.49,
        0.40,
        r"joint priority $S_\alpha$",
        color=GREEN,
        rotation=39,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.3),
        zorder=5,
    )
    ax.text(
        0.97,
        0.95,
        "high sealed\npriority",
        ha="right",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=GREEN,
    )

    ax.text(
        0.03,
        0.96,
        r"$\widehat q_G$: can this retained loss move?",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=BLUE,
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.90,
        r"$\widehat q_H$: is it exposed to this request?",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=ORANGE,
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.82,
        r"$S_\alpha=(1-\alpha)\widetilde q_G+\alpha\widetilde q_H$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK,
    )

    ax.text(0.04, 0.05, "low priority", fontsize=6.8, color=GRAY)
    ax.text(
        0.96,
        0.05,
        "exposed, but stable",
        fontsize=6.8,
        color=GRAY,
        ha="right",
    )
    ax.text(
        0.04,
        0.56,
        "fragile, but unrelated",
        fontsize=6.8,
        color=GRAY,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        r"request exposure $\widehat q_H$   low $\longrightarrow$ high",
        fontsize=8.2,
        color=ORANGE,
        labelpad=7,
    )
    ax.set_ylabel(
        r"loss-shake susceptibility $\widehat q_G$   low $\longrightarrow$ high",
        fontsize=8.2,
        color=BLUE,
        labelpad=8,
    )
    for spine in ax.spines.values():
        spine.set_color("#AFAFAF")
        spine.set_linewidth(0.8)


def _prediction_panel(ax) -> None:
    _panel_label(
        ax,
        "B",
        "PREDICT: prospective ordering",
        r"$\widehat\alpha_{\rm pred}$ fixed before audit outcomes",
    )
    x = np.linspace(0.06, 0.94, 14)
    heights = np.array(
        [0.12, 0.19, 0.17, 0.25, 0.22, 0.33, 0.30, 0.43, 0.38, 0.52, 0.60, 0.56, 0.73, 0.84]
    )
    colors = [GRAY] * 10 + [ORANGE] * 4
    for xi, yi, color in zip(x, heights, colors):
        ax.plot([xi, xi], [0.08, yi], color=color, lw=1.2, alpha=0.75)
        ax.scatter([xi], [yi], s=24, color=color, zorder=3)
    ax.plot([0.04, 0.96], [0.08, 0.08], color=INK, lw=0.9)
    ax.annotate(
        "",
        xy=(0.96, 0.035),
        xytext=(0.04, 0.035),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3),
    )
    ax.text(
        0.5,
        -0.02,
        r"sealed $S_{\widehat\alpha_{\rm pred}}$ rank   low $\longrightarrow$ high",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.6,
        color=GREEN,
    )
    ax.plot([x[-4] - 0.025, x[-1] + 0.025], [0.93, 0.93], color=ORANGE, lw=1.1)
    ax.plot([x[-4] - 0.025] * 2, [0.90, 0.96], color=ORANGE, lw=1.1)
    ax.plot([x[-1] + 0.025] * 2, [0.90, 0.96], color=ORANGE, lw=1.1)
    ax.text(
        (x[-4] + x[-1]) / 2,
        0.965,
        "later damage tail",
        ha="center",
        va="bottom",
        color=ORANGE,
        fontsize=7.2,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.82,
        "tests prospective ordering\nand harmful-tail recall",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=INK,
    )
    ax.text(
        0.98,
        0.13,
        "audit behaviors only",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=GRAY,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _selector_row(ax, y: float, label: str, chosen: set[int], color: str) -> None:
    xs = np.linspace(0.29, 0.65, 10)
    ax.text(0.02, y, label, ha="left", va="center", fontsize=7.3, color=INK)
    for i, x in enumerate(xs):
        active = i in chosen
        ax.scatter(
            [x],
            [y],
            s=28 if active else 20,
            fc=color if active else "white",
            ec=color if active else "#BDBDBD",
            lw=1.0,
            zorder=3,
        )


def _protection_panel(ax) -> None:
    _panel_label(
        ax,
        "C",
        "PROTECT: constrained allocation",
        r"$\widehat\alpha_{\rm prot}$ fixed; matched operator and budget",
    )
    _selector_row(ax, 0.78, "joint profile", {6, 7, 8, 9}, GREEN)
    _selector_row(ax, 0.60, r"$\widehat q_G$ only", {1, 4, 7, 9}, BLUE)
    _selector_row(ax, 0.42, r"$\widehat q_H$ only", {3, 6, 8, 9}, ORANGE)
    _selector_row(ax, 0.24, "random", {0, 3, 5, 8}, GRAY)
    ax.text(
        0.47,
        0.84,
        r"exactly the same $K_p$",
        ha="center",
        va="center",
        fontsize=6.8,
        color=GRAY,
    )

    box = FancyBboxPatch(
        (0.72, 0.12),
        0.27,
        0.73,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        fc="#F6F6F6",
        ec="#B8B8B8",
        lw=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        0.855,
        0.70,
        "EVALUATE\nheld-out retain",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.855,
        0.45,
        "mean · tail CVaR\nnative retain",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
        linespacing=1.35,
    )
    ax.text(
        0.855,
        0.23,
        "forget + utility",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=ORANGE,
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(0.715, 0.50),
        xytext=(0.68, 0.50),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0),
    )
    ax.text(
        0.02,
        0.04,
        "tests decision value; prediction alone does not guarantee protection",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color=GRAY,
        style="italic",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _figure_arrow(fig, start: tuple[float, float], end: tuple[float, float], label: str) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=11,
        lw=1.25,
        color=GREEN,
        connectionstyle="arc3,rad=0.0",
    )
    fig.add_artist(arrow)
    x = (start[0] + end[0]) / 2
    y = (start[1] + end[1]) / 2
    fig.text(
        x,
        y + 0.016,
        label,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=GREEN,
        bbox=dict(fc="white", ec="none", alpha=0.94, pad=0.8),
    )


def render(png_out: Path, pdf_out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )
    # Native size is close to ACM's two-column text width, so labels remain
    # legible without a large down-scaling in \includegraphics.
    fig = plt.figure(figsize=(8.0, 3.8), facecolor="white")
    profile = fig.add_axes([0.055, 0.17, 0.39, 0.55])
    prediction = fig.add_axes([0.56, 0.56, 0.405, 0.18])
    protection = fig.add_axes([0.56, 0.17, 0.405, 0.20])

    fig.text(
        0.055,
        0.975,
        "Why retained risk requires both sensitivity and exposure",
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.900,
        "One profile, two independent tests: predict future damage; allocate a fixed repair budget.",
        ha="left",
        va="top",
        fontsize=7.8,
        color=GRAY,
    )

    _profile_panel(profile)
    _prediction_panel(prediction)
    _protection_panel(protection)
    _figure_arrow(
        fig,
        (0.46, 0.54),
        (0.545, 0.63),
        "audit rank",
    )
    _figure_arrow(
        fig,
        (0.46, 0.34),
        (0.545, 0.26),
        r"Top-$K_p$ pool",
    )

    fig.text(
        0.51,
        0.065,
        "SCHEMATIC — candidates illustrate the estimands, not experimental results; quantitative evidence is reported in Tables 1–2.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=GRAY,
        style="italic",
    )

    png_out.parent.mkdir(parents=True, exist_ok=True)
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(
        pdf_out,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "plot_profile_two_uses.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--png-out",
        default="docs/figures/fig2_profile_two_uses.png",
    )
    parser.add_argument(
        "--pdf-out",
        default="paper/figures/fig2_profile_two_uses.pdf",
    )
    args = parser.parse_args()
    png_out = ROOT / args.png_out
    pdf_out = ROOT / args.pdf_out
    render(png_out, pdf_out)
    print(f"wrote png: {png_out}")
    print(f"wrote pdf: {pdf_out}")


if __name__ == "__main__":
    main()
