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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
FAMILY = {"grad_norm": "gradient", "fd_norm": "gradient", "knn_feature": "representation",
          "knn_embed": "representation", "knn_lexical": "representation",
          "fd": "alignment", "random_rank": "control"}
COL_ORDER = ["graddiff", "npo", "rmu"]
CHANNEL = {"graddiff": "loss-gradient", "npo": "loss-gradient", "rmu": "representation"}
COL_LABEL = {"graddiff": "GradDiff", "npo": "NPO*", "rmu": "RMU"}


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

    rows = [r for r in ROW_ORDER if r in rho]
    cols = [c for c in COL_ORDER if c in next(iter(rho.values()))]
    M = np.array([[rho[r].get(c, np.nan) for c in cols] for r in rows])

    fig = plt.figure(figsize=(9.2, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])

    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(cols)), [COL_LABEL.get(c, c) for c in cols])
    ax.set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.33 else "black", fontsize=9)
    # channel divider (loss-gradient | representation)
    n_lg = sum(1 for c in cols if CHANNEL[c] == "loss-gradient")
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

    # (B) interaction bars
    axb = fig.add_subplot(gs[0, 1])
    pairs = [("grad_norm", "knn_feature"), ("fd_norm", "knn_feature")]
    labels, deltas = [], []
    for gp, rp in pairs:
        if gp in rho and rp in rho and "graddiff" in cols and "rmu" in cols:
            d = ((rho[gp]["graddiff"] - rho[rp]["graddiff"])
                 - (rho[gp]["rmu"] - rho[rp]["rmu"]))
            labels.append(f"{gp}\nvs {rp}")
            deltas.append(d)
    y = np.arange(len(labels))
    axb.barh(y, deltas, color=["#2166ac", "#4393c3"][: len(labels)])
    axb.axvline(0, color="k", lw=0.8)
    axb.set_yticks(y, labels, fontsize=8)
    axb.set_xlabel("interaction  $\\Delta$", fontsize=9)
    axb.set_title("(B)  channel matching", fontsize=10, loc="left")
    for yi, d in zip(y, deltas):
        axb.text(d + 0.02, yi, f"{d:+.2f}", va="center", fontsize=9)
    axb.set_xlim(-0.1, max(deltas) * 1.35 if deltas else 1)
    axb.invert_yaxis()

    fig.suptitle(a.title, fontsize=10, y=1.02)
    fig.text(0.5, -0.04, "* NPO barely reached the forgetting criterion (near-static regime): weak signal, not a channel.",
             ha="center", fontsize=7.5, color="0.4")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
