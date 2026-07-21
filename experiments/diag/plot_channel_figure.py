"""Main figure: channel x predictor-family susceptibility.

(A) heatmap of Spearman rho(predictor, objective), rows grouped by predictor
    family, columns grouped by declared damage channel; the eye should see the
    gradient family light up on loss-gradient objectives and go dark on the
    representation objective, while the representation family lights up on RMU.
(B) the objective x family INTERACTION delta (difference-in-differences): the
    gradient probe's rank-advantage over the representation probe, on a
    loss-gradient objective minus on the representation objective.

Reads a channel_report.csv (from channel_report.py) via --report, else renders
the built-in DRAFT from the measured 7B single-request numbers.

  python experiments/diag/plot_channel_figure.py --report runs/gate_.../channel_report.csv --out fig.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rsus.analysis.channels import DECLARED_CHANNEL, PREDICTOR_FAMILY  # noqa: E402

CHAN_DISPLAY = {"loss_gradient": "loss-gradient", "representation": "representation"}
OBJ_LABEL = {"ga": "GA", "graddiff": "GradDiff", "npo": "NPO", "simnpo": "SimNPO",
             "idkdpo": "IdkDPO", "gru": "GRU", "rmu": "RMU"}

# Measured 7B, single request tofu-a180 (fd_norm shown at eta=3e-4, pre-fix;
# corrected eta=3e-3 re-score makes fd_norm ~ grad_norm per the fidelity gate).
DRAFT_RHO = {
    "grad_norm":   {"graddiff": 0.309, "npo": 0.173, "rmu": -0.122},
    "fd_norm":     {"graddiff": 0.359, "npo": 0.116, "rmu": 0.124},
    "knn_feature": {"graddiff": 0.378, "npo": 0.075, "rmu": 0.547},
    "knn_embed":   {"graddiff": 0.278, "npo": 0.133, "rmu": 0.353},
    "knn_lexical": {"graddiff": 0.264, "npo": 0.089, "rmu": 0.529},
    "fd":          {"graddiff": -0.071, "npo": -0.040, "rmu": -0.126},
    "random_rank": {"graddiff": 0.057, "npo": 0.050, "rmu": -0.010},
}
ROW_ORDER = ["grad_norm", "fd_norm", "knn_feature", "knn_embed", "knn_lexical", "fd", "random_rank"]
FAMILY = PREDICTOR_FAMILY


def order_columns(objs: list[str]) -> list[str]:
    """Objectives grouped by declared channel (loss_gradient first), name-sorted."""
    rank = {"loss_gradient": 0, "representation": 1}
    return sorted(objs, key=lambda o: (rank.get(DECLARED_CHANNEL.get(o, "z"), 2), o))


def load_report(path: Path) -> dict[str, dict[str, float]]:
    rho: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rho.setdefault(r["predictor"], {})[r["objective"]] = float(r["rho"])
    return rho


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "runs" / "channel_main_figure.png"))
    p.add_argument("--title", default="Channel-conditioned retain susceptibility (7B, tofu-a180, single request)")
    a = p.parse_args()
    rho = load_report(Path(a.report)) if a.report else DRAFT_RHO

    rows = [r for r in ROW_ORDER if r in rho] + [r for r in rho if r not in ROW_ORDER]
    all_objs = sorted({o for d in rho.values() for o in d})
    cols = order_columns(all_objs)
    M = np.array([[rho[r].get(c, np.nan) for c in cols] for r in rows])

    fig = plt.figure(figsize=(9.2, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])

    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(cols)), [OBJ_LABEL.get(c, c) for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.33 else "black", fontsize=9)
    # channel divider (loss-gradient | representation)
    n_lg = sum(1 for c in cols if DECLARED_CHANNEL.get(c) == "loss_gradient")
    ax.axvline(n_lg - 0.5, color="k", lw=1.5)
    # family group brackets
    fam_bounds, prev, start = [], None, 0
    for idx, r in enumerate(rows + [None]):
        fam = FAMILY.get(r) if r else None
        if fam != prev:
            if prev is not None:
                fam_bounds.append((start, idx - 1, prev))
            start, prev = idx, fam
    for s, e, fam in fam_bounds:
        ax.axhline(e + 0.5, color="0.6", lw=0.8, ls=":")
        if fam in ("gradient", "representation"):  # label only the two headline families
            ax.text(-1.7, (s + e) / 2, fam, rotation=90, va="center", ha="center",
                    fontsize=8.5, color="0.3", fontweight="bold")
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.text((n_lg - 1) / 2, -0.9, "loss-gradient channel", ha="center", fontsize=9, color="0.25")
    ax.text(n_lg + (len(cols) - n_lg - 1) / 2, -0.9, "representation", ha="center", fontsize=9, color="0.25")
    ax.set_title("(A)  rho(predictor, objective)", fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Spearman rho", fontsize=8)

    # (B) crossover: each headline probe's mean rho on each objective channel.
    # The lines CROSS -> the winning probe depends on the channel (= interaction).
    # Categorical colours (Okabe-Ito), NOT the diverging heatmap scale.
    axb = fig.add_subplot(gs[0, 1])
    HP = {"gradient": ("fd_norm", "#E69F00"), "representation": ("knn_feature", "#009E73")}
    chan_order = ["loss_gradient", "representation"]
    chan_objs = {ch: [c for c in cols if DECLARED_CHANNEL.get(c) == ch] for ch in chan_order}
    xpos = np.arange(len(chan_order))
    for fam, (probe, col) in HP.items():
        if probe not in rho:
            continue
        ys = [np.mean([rho[probe][o] for o in chan_objs[ch] if o in rho[probe]])
              if chan_objs[ch] else np.nan for ch in chan_order]
        axb.plot(xpos, ys, "o-", color=col, lw=2.2, ms=7, label=f"{probe}\n({fam} probe)")
        for x, yv in zip(xpos, ys):
            axb.text(x, yv + 0.03, f"{yv:.2f}", ha="center", fontsize=8, color=col)
    axb.axhline(0, color="0.7", lw=0.8, ls=":")
    axb.set_xticks(xpos, ["loss-\ngradient", "represen-\ntation"], fontsize=8.5)
    axb.set_xlim(-0.35, len(chan_order) - 0.65)
    axb.set_ylabel("mean $\\rho$ (probe, objective)", fontsize=8.5)
    axb.set_title("(B)  which probe wins by channel", fontsize=10, loc="left")
    axb.legend(fontsize=7, loc="upper center", frameon=False)
    axb.set_xlabel("objective channel", fontsize=8.5)

    fig.suptitle(a.title, fontsize=10, y=1.02)
    if "npo" in cols:
        fig.text(0.5, -0.04, "* NPO barely reached the forgetting criterion (near-static regime): "
                 "weak signal, not a channel.", ha="center", fontsize=7.5, color="0.4")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
