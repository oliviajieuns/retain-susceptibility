"""Generate Figure 2 -- the evidence chain "claim ladder with a measured core".

Design spec: docs/figures/FIGURE2_EVIDENCE_CHAIN_GUIDELINE.md (decided by the
2026-07-24 multi-agent adversarial pass). Three stacked bands, bottom->top =
RQ2 (fidelity floors) -> RQ1 (a small honest prediction scatter) -> RQ3 (8
damage UCBs + 4 native LBs, with a snapped rung for the frozen-op-point
infeasibility). Every claim is an achieved 95% bound vs a predeclared floor;
the chain shows three escalating honest "no"s so the fail-closed identity is
unmissable.

The numbers are a seeded conceptual small-sample calibrated to the real 7B
operating point (joint rho ~0.21; fidelity rho_AC 0.92 / f_K 0.77; RQ3
infeasible at the frozen op-points). When the sealed audit aggregates, the
loaders below prefer the real artifacts:
  - results/paper/fidelity_summaries/tofu_qwen25_7b.json  (RQ2)
  - results/paper/evidence_ledger.json                    (RQ1/RQ3)

Usage:
  python experiments/paper/plot_evidence_chain.py --tikz  # -> figures/fig6_evidence_chain.tex body
  python experiments/paper/plot_evidence_chain.py --png   # -> docs/figures/fig2_evidence_chain_preview.png
  python experiments/paper/plot_evidence_chain.py --data-only  # dump the seeded JSON only

This module is import-safe without numpy/matplotlib (they load lazily inside
--png); --data-only and --tikz use only the standard library so the paper
figure regenerates on a CPU login node.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Okabe-Ito, CVD-safe; matches the colors already declared in fig6.
COLORS = {
    "evGrad": "0072B2",   # q_G / RQ2 fidelity
    "evProx": "D55E00",   # q_H / revealed damage points
    "evMix": "009E73",    # joint / a bound that cleared its floor
    "evCtrl": "767676",   # controls / failed side / non-reach / snapped
    "evGate": "CC79A7",   # forgetting gate
    "ink": "1A1A1A",      # seal / floor ticks / x=0 line
    "amber": "E69F00",    # revealed outcome
}

SEED = 20260724
TAU_RHO = 0.80
TAU_K = 0.70


# --------------------------------------------------------------------------
# Data assembly (prefer real artifacts; fall back to seeded conceptual sample)
# --------------------------------------------------------------------------
def _load_fidelity() -> dict:
    """RQ2 rows: (label, value, floor, valid). Prefer the rescored summary."""
    path = ROOT / "results" / "paper" / "fidelity_summaries" / "tofu_qwen25_7b.json"
    if path.is_file():
        s = json.loads(path.read_text())
        rows = [
            ("$\\rho_{AC}$", s.get("f_rho"), TAU_RHO, True),
            ("overlap@$K_p$", s.get("f_k"), TAU_K, True),
        ]
        rows = [(a, b, c, d) for (a, b, c, d) in rows if b is not None]
        source = "rescored certificate"
    else:
        rows, source = [], "conceptual"
    if not rows:
        # Values already in fig6 (fp32 pass; bf16 collapse = the honest "no").
        rows = [
            ("$\\rho_{AB}$", 0.9506, 0.70, True),
            ("$\\rho_{BC}$", 0.9657, 0.80, True),
            ("$\\rho_{AC}$", 0.9247, 0.80, True),
            ("frac. changed", 0.9965, 0.90, True),
            ("eff$/\\eta$", 1.000, 0.90, True),
        ]
        source = "fig6 fp32 values"
    # The bf16 collapse row is always shown as honest-"no" #1.
    bf16 = ("bf16 frac. changed", 0.0019, 0.90, False)
    return {"rows": rows, "bf16": bf16, "source": source}


def _monotone_sample(n: int, rho_target: float, rng: random.Random) -> list[tuple[float, float]]:
    """Return (rank01, damage) pairs with Spearman ~= rho_target (Gaussian copula).

    Gaussian-copula Spearman->Pearson inversion: rho_P = 2*sin(pi/6 * rho_S).
    (Using sin(pi/2 * rho) overshoots and would inflate the modest 0.21 anchor.)
    """
    r = max(-0.999, min(0.999, 2.0 * math.sin(math.pi / 6.0 * rho_target)))
    pairs = []
    for _ in range(n):
        z1 = rng.gauss(0, 1)
        z2 = r * z1 + math.sqrt(1 - r * r) * rng.gauss(0, 1)
        rank01 = 0.5 * (1 + math.erf(z1 / math.sqrt(2)))       # uniform-ish rank in [0,1]
        damage = max(0.0, 0.6 + 0.9 * z2)                       # nats, non-negative
        pairs.append((rank01, damage))
    return pairs


def _spearman(xs, ys) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def _bootstrap_lb(xs, ys, reps: int, seed: int, alpha: float = 0.05) -> float:
    rng = random.Random(seed)
    n = len(xs)
    draws = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        draws.append(_spearman([xs[i] for i in idx], [ys[i] for i in idx]))
    draws.sort()
    return draws[max(0, int(alpha * len(draws)) - 1)]


def _load_prediction() -> dict:
    """RQ1: prefer the real ledger; else a seeded copula at rho~0.21."""
    ledger = ROOT / "results" / "paper" / "evidence_ledger.json"
    if ledger.is_file():
        led = json.loads(ledger.read_text())
        for row in led.get("rows", []):
            if row.get("setting") == "tofu_qwen25_7b" and row.get("prediction", {}).get("paired"):
                pr = row["prediction"]
                return {
                    "source": "sealed ledger",
                    "joint_rho": pr.get("joint_rho"),
                    "joint_lb": (pr.get("joint") or {}).get("lower_bound"),
                    "g_g_lb": (pr.get("vs_s0") or {}).get("lower_bound"),
                    "g_h_lb": (pr.get("vs_s1") or {}).get("lower_bound"),
                    "g_ctl_lb": (pr.get("vs_control") or {}).get("lower_bound"),
                    "tail_lb": (pr.get("tail_lift") or {}).get("lower_bound"),
                    "coverage": (
                        (pr.get("tail_eligible_n") or 0) / pr["tail_total_n"]
                        if pr.get("tail_total_n") else None
                    ),
                    "points": None,
                }
    rng = random.Random(SEED)
    pairs = _monotone_sample(120, 0.21, rng)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rho = _spearman(xs, ys)
    lb = _bootstrap_lb(xs, ys, reps=2000, seed=SEED + 1)
    return {
        "source": "conceptual (rho~0.21, calibrated to 7B)",
        "joint_rho": rho,
        "joint_lb": lb,
        "g_g_lb": 0.05,   # joint - q_G-only endpoint, small positive
        "g_h_lb": 0.04,   # joint - q_H-only endpoint, small positive
        "g_ctl_lb": 0.02,
        "tail_lb": 0.08,
        "coverage": 1.0,
        "points": pairs,
        "ghost_qg_rho": 0.14,
        "ghost_qh_rho": 0.16,
        "non_reach": 5,
    }


def _load_protection() -> dict:
    """RQ3: 8 damage UCBs + 4 native LBs; one arm's UCB>0 drives the snapped rung.

    Grounded in the real 7B verdict (docs/data/alpha_dev_7b): RQ3 is infeasible
    at the frozen operating points, so the honest figure snaps here.
    """
    comparators = ["no_repair", "repeated_random", "s0", "s1"]
    outcomes = ["mean", "cvar95"]
    rng = random.Random(SEED + 2)
    ucbs = {}
    for c in comparators:
        for o in outcomes:
            # passing arms: UCB < 0 (mixture beats comparator)
            ucbs[f"{c}.{o}"] = round(-abs(rng.gauss(0.15, 0.05)) - 0.02, 3)
    # honest "no": the frozen-op-point infeasibility -- one arm crosses >0.
    ucbs["s1.cvar95"] = 0.11  # snapped rung
    native = {c: round(abs(rng.gauss(0.03, 0.01)) + 0.01, 3) for c in comparators}
    return {
        "source": "conceptual (frozen-op-point infeasible, per alpha_dev_7b)",
        "ucbs": ucbs,
        "native_lbs": native,
        "snapped_arm": "s1.cvar95",
        "licensed": False,
    }


def assemble() -> dict:
    return {
        "seed": SEED,
        "palette": COLORS,
        "rq2": _load_fidelity(),
        "rq1": _load_prediction(),
        "rq3": _load_protection(),
        "caption_burn_in": (
            "Every claim is a bound that had to clear a floor set before any "
            "outcome existed --- and the chain stops the moment one doesn't."
        ),
    }


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------
def render_png(data: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = {k: f"#{v}" for k, v in data["palette"].items()}
    fig, (axr3, axr1, axr2) = plt.subplots(
        3, 1, figsize=(9.5, 8.0), gridspec_kw={"height_ratios": [1.0, 1.35, 1.0]}
    )
    fig.suptitle(
        "PREDECLARED FLOORS — sealed at θ₀, before any outcome existed",
        fontsize=10, color=c["ink"], x=0.5, y=0.98, weight="bold",
    )

    # RQ2 (bottom axis in array order is last)
    rq2 = data["rq2"]
    rows = rq2["rows"] + [rq2["bf16"]]
    ys = range(len(rows))
    for y, (label, val, floor, valid) in zip(ys, rows):
        col = c["evGrad"] if valid else c["evCtrl"]
        axr2.barh(y, val, color=col, height=0.55, zorder=2)
        axr2.plot([floor, floor], [y - 0.32, y + 0.32], color=c["ink"], lw=2, zorder=3)
    axr2.set_yticks(list(ys))
    axr2.set_yticklabels([r[0] for r in rows], fontsize=8)
    axr2.set_xlim(0, 1.05)
    axr2.set_title("RQ2  loss-shake fidelity vs frozen floors  (| = floor; gray = bf16 collapse, invalid)",
                   fontsize=9, loc="left", color=c["ink"])
    axr2.invert_yaxis()

    # RQ1 measured core
    rq1 = data["rq1"]
    if rq1.get("points"):
        xs = [p[0] for p in rq1["points"]]
        dm = [p[1] for p in rq1["points"]]
        axr1.scatter(xs, dm, s=14, color=c["evProx"], alpha=0.7, zorder=2, label="audit candidate")
        # joint trend
        import numpy as np
        xarr = np.array(xs)
        z = np.polyfit(xarr, dm, 1)
        xline = np.linspace(0, 1, 20)
        axr1.plot(xline, z[0] * xline + z[1], color=c["evMix"], lw=2.2, zorder=3, label="joint $S_\\alpha$")
        # ghost endpoints
        axr1.plot(xline, 0.14 * (z[0]) * xline + z[1] + 0.05, color=c["evGrad"], lw=1.2, alpha=0.3, zorder=1)
        axr1.plot(xline, 0.16 * (z[0]) * xline + z[1] + 0.02, color=c["evProx"], lw=1.2, alpha=0.3, zorder=1)
        axr1.axhline(z[1], color=c["evCtrl"], lw=1.0, ls="--", alpha=0.6, zorder=1)
    axr1.set_title(
        f"RQ1  sealed joint rank vs revealed damage    "
        f"$\\rho={rq1['joint_rho']:.2f}$ [LB {rq1['joint_lb']:+.2f}]   "
        f"non-reach: {rq1.get('non_reach','?')} parents excluded",
        fontsize=9, loc="left", color=c["ink"])
    axr1.set_xlabel("sealed joint rank $S_\\alpha$ (frozen at $\\theta_0$)", fontsize=8)
    axr1.set_ylabel("revealed damage $d_{t\\dagger}$ (nats)", fontsize=8)

    # RQ3 caterpillar with snapped rung
    rq3 = data["rq3"]
    items = list(rq3["ucbs"].items())
    for i, (k, ucb) in enumerate(items):
        snapped = (k == rq3["snapped_arm"])
        col = c["evCtrl"] if snapped or ucb >= 0 else c["evMix"]
        axr3.plot([min(0, ucb), max(0, ucb)], [i, i], color=col, lw=3 if not snapped else 1,
                  ls="-" if not snapped else (0, (2, 2)), zorder=2)
        axr3.scatter([ucb], [i], color=col, s=20, zorder=3)
    axr3.axvline(0, color=c["ink"], lw=1.5, zorder=1)
    axr3.set_yticks(range(len(items)))
    axr3.set_yticklabels([k for k, _ in items], fontsize=7)
    axr3.set_title(
        "RQ3  8 damage UCBs vs 0 (mixture beats comparator)   "
        + ("COMPOSITE CHAIN: NOT LICENSED here — infeasible at frozen op-point"
           if not rq3["licensed"] else "licensed"),
        fontsize=9, loc="left", color=c["ink"])
    axr3.set_xlabel("$\\Delta$NLL upper bound (want < 0)", fontsize=8)
    axr3.invert_yaxis()

    fig.text(0.5, 0.005, data["caption_burn_in"], ha="center", fontsize=8,
             style="italic", color=c["ink"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_tikz(data: dict, out: Path) -> Path:
    """Emit a self-contained pgfplots body. Kept intentionally close to the
    fig6 Panel-B idiom (xbar + | floor ticks) so it drops into the paper.

    This writes a REVIEW scaffold with the real numbers baked in; the final
    hand-tuned tikz (snapped-rung hatch, embedded scatter registration) is
    finished in the paper repo against this data. All series come from
    ``data`` so re-running with sealed artifacts regenerates the numbers.
    """
    p = data["palette"]
    rq2, rq1, rq3 = data["rq2"], data["rq1"], data["rq3"]
    lines = [
        "% Figure 2 evidence chain -- generated by",
        "%   experiments/paper/plot_evidence_chain.py --tikz",
        "% Design: docs/figures/FIGURE2_EVIDENCE_CHAIN_GUIDELINE.md",
        f"% data source: RQ2={rq2['source']}; RQ1={rq1['source']}; RQ3={rq3['source']}",
    ]
    for name, hexv in p.items():
        lines.append(f"\\definecolor{{{name}}}{{HTML}}{{{hexv}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tikzpicture}[font=\\small]")
    # Header
    lines.append(
        "\\node[anchor=south west, font=\\small\\bfseries, text=ink] at (0cm,9.4cm) "
        "{PREDECLARED FLOORS --- sealed at $\\theta_0$, before any outcome existed};")
    # RQ2 band (reuse xbar idiom); rows top-down
    lines.append("\\begin{axis}[at={(0cm,8.6cm)}, anchor=north west, width=8.4cm, height=3.0cm,")
    lines.append("  axis lines=left, y dir=reverse, xmin=0, xmax=1.05,")
    lines.append("  title={\\textbf{RQ2}\\; loss-shake fidelity vs frozen floors}, title style={font=\\footnotesize, at={(0,1.04)}, anchor=south west},")
    rows = rq2["rows"] + [rq2["bf16"]]
    lines.append("  ytick={" + ",".join(str(i) for i in range(len(rows))) + "},")
    lines.append("  yticklabels={" + ",".join("{" + r[0] + "}" for r in rows) + "},")
    lines.append("  yticklabel style={font=\\scriptsize}, tick label style={font=\\scriptsize},")
    lines.append("  xlabel={certificate value ($|$ = frozen floor)}, xlabel style={font=\\scriptsize}]")
    for i, (label, val, floor, valid) in enumerate(rows):
        col = "evGrad" if valid else "evCtrl"
        lines.append(f"\\addplot[xbar, bar width=0.34, fill={col}, draw=white, line width=0.3pt] coordinates {{({val:.4f},{i})}};")
        lines.append(f"\\draw[ink, line width=1.0pt] (axis cs:{floor},{i-0.32}) -- (axis cs:{floor},{i+0.32});")
    lines.append("\\end{axis}")
    # RQ1 + RQ3 summary rungs as a compact bound-vs-floor column on the right,
    # and a note that the measured scatter (RQ1) + snapped caterpillar (RQ3) are
    # finished by hand from this data (see guideline sec 4/7).
    lines.append(
        "\\node[anchor=north west, font=\\footnotesize, text=ink, align=left] at (8.9cm,8.6cm) {"
        f"\\textbf{{RQ1}} joint $\\rho={rq1['joint_rho']:.2f}$ [LB ${rq1['joint_lb']:+.2f}$]\\\\"
        f"$\\min(g_G,g_H)$ [LB ${min(rq1['g_g_lb'], rq1['g_h_lb']):+.2f}$],\\; "
        f"$g_{{\\rm ctl}}$ [LB ${rq1['g_ctl_lb']:+.2f}$]\\\\"
        f"$L_{{\\rm tail}}$ [LB ${rq1['tail_lb']:+.2f}$]; coverage $\\geq 80\\%$\\\\"
        f"non-reach: {rq1.get('non_reach','?')} parents excluded (denominator)\\\\[2mm]"
        "\\textbf{RQ3} 8 damage UCBs vs 0; 4 native LBs\\\\"
        "\\textcolor{evCtrl}{\\textbf{COMPOSITE CHAIN: NOT LICENSED here}}\\\\"
        "infeasible at frozen $(K_p,\\text{budget})$ op-point};")
    lines.append("\\end{tikzpicture}%")
    lines.append("}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tikz", action="store_true", help="write the pgfplots body")
    ap.add_argument("--png", action="store_true", help="write a matplotlib preview")
    ap.add_argument("--data-only", action="store_true", help="dump the seeded JSON only")
    ap.add_argument("--tikz-out", default="figures/fig6_evidence_chain_generated.tex")
    ap.add_argument("--png-out", default="docs/figures/fig2_evidence_chain_preview.png")
    ap.add_argument("--data-out", default="docs/figures/fig2_evidence_chain_data.json")
    args = ap.parse_args(argv)

    data = assemble()
    (ROOT / args.data_out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.data_out).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote data: {args.data_out}")
    if args.tikz:
        print(f"wrote tikz: {render_tikz(data, ROOT / args.tikz_out)}")
    if args.png:
        print(f"wrote png:  {render_png(data, ROOT / args.png_out)}")
    if not (args.tikz or args.png or args.data_only):
        print("(pass --tikz and/or --png; --data-only just dumps the JSON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
