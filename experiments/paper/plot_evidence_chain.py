"""Render Figure 2: the evidence chain for the sealed evaluation framework.

Panel A  absolute and incremental prediction on repair-held-out behaviors,
         from one or more audit ``pooled_channel_report.csv`` files
         (experiments/channel_matrix/aggregate.py).
Panel B  loss-shake fidelity certificates against their frozen thresholds
         (fd-fidelity-certificate-v1 JSON), plus the joint-profile value of
         the deployed mixture beyond the frozen single-probe priors
         (``alpha_protection_contrasts.csv`` comparators alpha0.0/alpha1.0).
Panel C  constraint-matched fixed-budget protection: paired CVaR damage
         contrasts against allocation comparators (random / none /
         exact_grad_norm) plus feasibility and native retained behavior at
         the deployed alpha (``alpha_protection_curve.csv``).

Every input is optional under ``--allow-partial`` so the figure can be
rendered incrementally while audit waves are still draining; a missing panel
shows an explicit "pending" placeholder instead of silently vanishing.

Rendering is container-side (CPU, matplotlib Agg). Cluster runs only produce
the CSV/JSON inputs.

Example:
    python experiments/paper/plot_evidence_chain.py \
      --channel-report "TOFU-7B=runs/agg_7b/pooled_channel_report.csv" \
      --channel-report "TOFU-14B=runs/agg_14b/pooled_channel_report.csv" \
      --fidelity "TOFU-7B bf16=docs/data/fidelity/fd_fidelity_7b_bf16.json" \
      --protection "TOFU-7B=runs/alpha_agg_7b" \
      --allow-partial --out figures/fig2_evidence_chain.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (scripts/validate_palette.js: ALL CHECKS PASS).
C_GRAD = "#0072B2"   # gradient-family probe (fd_norm)
C_PROX = "#D55E00"   # proximity-family probe (knn_feature)
C_THIRD = "#009E73"  # deployed mixture / third series
C_CTRL = "#767676"   # frozen random-rank control (neutral, non-series)
INK = "#1a1a1a"
MUTED = "#5f5f5f"

OBJ_LABEL = {
    "ga": "GA",
    "graddiff": "GradDiff",
    "npo": "NPO",
    "simnpo": "SimNPO",
    "gru": "GRU",
    "idkdpo": "IDK-DPO",
    "rmu": "RMU",
    "repnoise": "RepNoise",
    "circuit_breakers": "CircuitBreakers",
}
PRED_LABEL = {
    "fd_norm": "loss-shake (fd_norm)",
    "grad_norm": "grad-norm",
    "knn_feature": "feature k-NN",
    "random_rank": "random control",
}
FIDELITY_METRICS = [
    ("rho_AB", r"$\rho_{AB}$"),
    ("rho_BC", r"$\rho_{BC}$"),
    ("rho_AC", r"$\rho_{AC}$"),
    ("frac_changed", "frac. changed"),
    ("eff_over_eta", r"eff/$\eta$"),
]
PRIOR_COMPARATORS = ("alpha0.0", "alpha1.0")
BUDGET_COMPARATORS = ("none", "random", "exact_grad_norm")
COMPARATOR_LABEL = {
    "alpha0.0": r"frozen gradient prior ($\alpha$=0)",
    "alpha1.0": r"frozen proximity prior ($\alpha$=1)",
    "none": "no protection",
    "random": "random allocation",
    "exact_grad_norm": "exact grad-norm",
}


def _parse_labeled(values: list[str], kind: str) -> list[tuple[str, Path]]:
    out = []
    for value in values:
        label, sep, path = value.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"--{kind} expects LABEL=PATH, got {value!r}")
        out.append((label, Path(path)))
    return out


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fnum(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float("nan")
    return float(value)


def _placeholder(ax, text: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9,
            color=MUTED, style="italic", transform=ax.transAxes, wrap=True)


# ---------------------------------------------------------------- panel A
def panel_a(ax_abs, ax_inc, reports: list[tuple[str, list[dict]]],
            predictors: list[str], control: str) -> None:
    """Absolute rho (95% CI) per core objective, and incremental rho over
    the frozen random-rank control, grouped by campaign."""
    groups = []  # (tick_label, {predictor: (rho, lo, hi)})
    for label, rows in reports:
        core = [r for r in rows if r.get("channel") in ("loss_gradient", "representation")]
        objectives = sorted({r["objective"] for r in core},
                            key=lambda o: list(OBJ_LABEL).index(o) if o in OBJ_LABEL else 99)
        # Core-only view: an aggregate CSV also carries stress objectives, but
        # the evidence chain reads the core roster of each campaign. Callers
        # pre-filter by passing a core-only CSV; here we keep every objective
        # present and rely on the campaign aggregates being core-scoped.
        for objective in objectives:
            cell = {}
            for predictor in predictors + [control]:
                match = [r for r in core
                         if r["objective"] == objective and r["predictor"] == predictor]
                if match:
                    cell[predictor] = (
                        _fnum(match[0], "rho"),
                        _fnum(match[0], "rho_lo"),
                        _fnum(match[0], "rho_hi"),
                    )
            if cell:
                groups.append((f"{OBJ_LABEL.get(objective, objective)}\n{label}", cell))

    if not groups:
        _placeholder(ax_abs, "Panel A pending:\nno pooled_channel_report.csv yet")
        ax_inc.set_axis_off()
        return

    xs = range(len(groups))
    offsets = {p: (i - (len(predictors) - 1) / 2) * 0.22
               for i, p in enumerate(predictors)}
    colors = {predictors[0]: C_GRAD}
    if len(predictors) > 1:
        colors[predictors[1]] = C_PROX
    for extra in predictors[2:]:
        colors[extra] = C_THIRD

    for predictor in predictors:
        pts = [(x, cell[predictor]) for x, (_, cell) in zip(xs, groups)
               if predictor in cell]
        if not pts:
            continue
        px = [x + offsets[predictor] for x, _ in pts]
        py = [v[0] for _, v in pts]
        err_lo = [max(0.0, v[0] - v[1]) if math.isfinite(v[1]) else 0.0 for _, v in pts]
        err_hi = [max(0.0, v[2] - v[0]) if math.isfinite(v[2]) else 0.0 for _, v in pts]
        ax_abs.errorbar(px, py, yerr=[err_lo, err_hi], fmt="o", ms=5,
                        color=colors[predictor], ecolor=colors[predictor],
                        elinewidth=1.2, capsize=2.5,
                        label=PRED_LABEL.get(predictor, predictor))
    ctrl_pts = [(x, cell[control][0]) for x, (_, cell) in zip(xs, groups)
                if control in cell]
    if ctrl_pts:
        ax_abs.plot([x for x, _ in ctrl_pts], [v for _, v in ctrl_pts],
                    "o", ms=5, mfc="none", mec=C_CTRL, mew=1.2,
                    label=PRED_LABEL.get(control, control))

    ax_abs.axhline(0.0, color=C_CTRL, lw=0.8, ls=":")
    ax_abs.set_ylabel(r"Spearman $\rho$ (score vs. realized $\Delta$NLL)")
    ax_abs.set_xticks(list(xs))
    ax_abs.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax_abs.legend(fontsize=7, frameon=False, loc="upper right")
    ax_abs.set_title("A  absolute and incremental prediction\n(repair-held-out behaviors)",
                     loc="left", fontsize=10, color=INK)

    # Incremental: paired delta over the frozen control, same pooled runs.
    for predictor in predictors:
        bx, by = [], []
        for x, (_, cell) in zip(xs, groups):
            if predictor in cell and control in cell:
                bx.append(x + offsets[predictor])
                by.append(cell[predictor][0] - cell[control][0])
        if bx:
            ax_inc.bar(bx, by, width=0.20, color=colors[predictor],
                       edgecolor="white", linewidth=0.5)
    ax_inc.axhline(0.0, color=C_CTRL, lw=0.8)
    ax_inc.set_ylabel(r"$\Delta\rho$ vs. control", fontsize=8)
    ax_inc.set_xticks(list(xs))
    ax_inc.set_xticklabels([g[0] for g in groups], fontsize=7.5)


# ---------------------------------------------------------------- panel B
def panel_b(ax_fid, ax_joint, certificates: list[tuple[str, dict]],
            contrasts: list[tuple[str, list[dict]]]) -> None:
    """Fidelity metrics vs frozen thresholds; deployed-mixture value beyond
    the frozen single-probe priors."""
    if certificates:
        n_metrics = len(FIDELITY_METRICS)
        height = 0.8 / max(1, len(certificates))
        for ci, (label, cert) in enumerate(certificates):
            metrics = cert.get("metrics", {})
            thresholds = cert.get("thresholds", {})
            ys, vals, cols = [], [], []
            for mi, (key, _) in enumerate(FIDELITY_METRICS):
                value = float(metrics.get(key, float("nan")))
                floor = float(thresholds.get(key, float("nan")))
                ys.append(mi + ci * height - 0.4 + height / 2)
                vals.append(value)
                ok = math.isfinite(value) and math.isfinite(floor) and value >= floor
                cols.append(C_GRAD if ok else C_PROX)
            ax_fid.barh(ys, vals, height=height * 0.9, color=cols,
                        edgecolor="white", linewidth=0.5)
            # Bar color encodes pass/fail against the floor, so certificate
            # identity is carried by an in-bar tag on the first row (rho_AB
            # is long for every realistic certificate), not a legend.
            ax_fid.text(0.02, ys[0], label, fontsize=6, color="white",
                        ha="left", va="center")
            for (key, _), y in zip(FIDELITY_METRICS, ys):
                floor = thresholds.get(key)
                if floor is not None:
                    ax_fid.plot([float(floor)] * 2,
                                [y - height * 0.45, y + height * 0.45],
                                color=INK, lw=1.4)
        ax_fid.set_yticks(range(n_metrics))
        ax_fid.set_yticklabels([lbl for _, lbl in FIDELITY_METRICS], fontsize=8)
        ax_fid.invert_yaxis()
        ax_fid.set_xlim(0, 1.05)
        ax_fid.set_xlabel("certificate value (| = frozen floor)", fontsize=8)
    else:
        _placeholder(ax_fid, "fidelity certificates pending")
    ax_fid.set_title("B  loss-shake fidelity and joint-profile value",
                     loc="left", fontsize=10, color=INK)

    rows = [(label, r) for label, rs in contrasts for r in rs
            if r.get("comparator") in PRIOR_COMPARATORS]
    if rows:
        _contrast_bars(ax_joint, rows,
                       xlabel="deployed mixture $-$ frozen prior,\nCVaR$_{05}$ $\\Delta$NLL (95% CI)")
    else:
        _placeholder(ax_joint, "joint-profile contrasts pending\n(alpha-audit not aggregated yet)")


def _contrast_bars(ax, rows: list[tuple[str, dict]], xlabel: str) -> None:
    ys = range(len(rows))
    labels, vals = [], []
    for label, row in rows:
        diff = _fnum(row, "mean_cvar_difference_deployed_minus_comparator")
        lo = _fnum(row, "ci95_lo")
        hi = _fnum(row, "ci95_hi")
        comparator = COMPARATOR_LABEL.get(row["comparator"], row["comparator"])
        labels.append(f"{label} {row.get('parent', '')}\nvs {comparator}")
        vals.append((diff, lo, hi))
    for y, (diff, lo, hi) in zip(ys, vals):
        color = C_THIRD if math.isfinite(diff) and diff < 0 else C_PROX
        ax.barh(y, diff, height=0.6, color=color, edgecolor="white", linewidth=0.5)
        if math.isfinite(lo) and math.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=INK, lw=1.2)
            ax.plot([lo, lo], [y - 0.15, y + 0.15], color=INK, lw=1.2)
            ax.plot([hi, hi], [y - 0.15, y + 0.15], color=INK, lw=1.2)
    ax.axvline(0.0, color=C_CTRL, lw=0.8, ls=":")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8)


# ---------------------------------------------------------------- panel C
def panel_c(ax_damage, ax_feas, contrasts: list[tuple[str, list[dict]]],
            curves: list[tuple[str, list[dict]]]) -> None:
    """Fixed-budget allocation: damage contrasts + feasibility and native
    retained behavior at the deployed alpha."""
    rows = [(label, r) for label, rs in contrasts for r in rs
            if r.get("comparator") in BUDGET_COMPARATORS]
    if rows:
        _contrast_bars(ax_damage, rows,
                       xlabel="deployed $-$ comparator, CVaR$_{05}$ $\\Delta$NLL (95% CI)")
    else:
        _placeholder(ax_damage, "Panel C pending:\nalpha-audit contrasts not aggregated yet")
    ax_damage.set_title("C  constraint-matched\nfixed-budget protection",
                        loc="left", fontsize=10, color=INK)

    points = []
    for label, rows_curve in curves:
        deployed = [r for r in rows_curve if int(_fnum(r, "deployed_model_count") or 0) > 0]
        for row in deployed:
            n_total = _fnum(row, "n_total")
            n_reach = _fnum(row, "n_reach")
            feas = n_reach / n_total if n_total else float("nan")
            points.append((
                f"{label} {row.get('parent', '')}\n" + r"$\hat\alpha$=" + f"{_fnum(row, 'alpha'):g}",
                feas,
                _fnum(row, "mean_utility_retention_reached"),
            ))
    if points:
        xs = range(len(points))
        ax_feas.bar([x - 0.18 for x in xs], [p[1] for p in points], width=0.32,
                    color=C_GRAD, edgecolor="white", linewidth=0.5,
                    label="feasibility (reach rate)")
        ax_feas.bar([x + 0.18 for x in xs], [p[2] for p in points], width=0.32,
                    color=C_THIRD, edgecolor="white", linewidth=0.5,
                    label="native retained behavior\n(utility retention)")
        ax_feas.axhline(1.0, color=C_CTRL, lw=0.8, ls=":")
        ax_feas.axhline(0.90, color=C_PROX, lw=0.8, ls="--")
        ax_feas.set_ylim(0, 1.1)
        ax_feas.set_xticks(list(xs))
        ax_feas.set_xticklabels([p[0] for p in points], fontsize=6.5)
        ax_feas.legend(fontsize=6.5, frameon=False, loc="lower right")
    else:
        _placeholder(ax_feas, "feasibility / retention pending")


# ---------------------------------------------------------------- driver
def load_inputs(args) -> dict:
    data = {"reports": [], "certificates": [], "contrasts": [], "curves": []}
    for label, path in _parse_labeled(args.channel_report or [], "channel-report"):
        if path.exists():
            data["reports"].append((label, _read_csv(path)))
        elif not args.allow_partial:
            raise SystemExit(f"missing channel report: {path}")
    for label, path in _parse_labeled(args.fidelity or [], "fidelity"):
        if path.exists():
            data["certificates"].append(
                (label, json.loads(path.read_text(encoding="utf-8"))))
        elif not args.allow_partial:
            raise SystemExit(f"missing fidelity certificate: {path}")
    for label, path in _parse_labeled(args.protection or [], "protection"):
        contrasts = path / "alpha_protection_contrasts.csv"
        curve = path / "alpha_protection_curve.csv"
        if contrasts.exists():
            data["contrasts"].append((label, _read_csv(contrasts)))
        elif not args.allow_partial:
            raise SystemExit(f"missing protection contrasts: {contrasts}")
        if curve.exists():
            data["curves"].append((label, _read_csv(curve)))
        elif not args.allow_partial:
            raise SystemExit(f"missing protection curve: {curve}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-report", action="append",
                        help="LABEL=path/to/pooled_channel_report.csv (repeatable)")
    parser.add_argument("--fidelity", action="append",
                        help="LABEL=path/to/certificate.json (repeatable)")
    parser.add_argument("--protection", action="append",
                        help="LABEL=dir containing alpha_protection_{curve,contrasts}.csv")
    parser.add_argument("--predictors", default="fd_norm,knn_feature",
                        help="comma-separated headline probes for panel A")
    parser.add_argument("--control", default="random_rank",
                        help="frozen control predictor for the incremental row")
    parser.add_argument("--allow-partial", action="store_true",
                        help="render whichever panels have inputs; placeholders otherwise")
    parser.add_argument("--out", required=True, help="output figure (.pdf/.png)")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    data = load_inputs(args)
    if not any(data.values()) and not args.allow_partial:
        raise SystemExit("no inputs given; pass --allow-partial to render placeholders")

    fig = plt.figure(figsize=(13.5, 4.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0],
                            height_ratios=[2.2, 1.0], hspace=0.45, wspace=0.42,
                            left=0.06, right=0.985, top=0.86, bottom=0.14)
    ax_a_abs = fig.add_subplot(grid[0, 0])
    ax_a_inc = fig.add_subplot(grid[1, 0], sharex=ax_a_abs)
    ax_b_fid = fig.add_subplot(grid[0, 1])
    ax_b_joint = fig.add_subplot(grid[1, 1])
    ax_c_damage = fig.add_subplot(grid[0, 2])
    ax_c_feas = fig.add_subplot(grid[1, 2])

    for ax in (ax_a_abs, ax_a_inc, ax_b_fid, ax_b_joint, ax_c_damage, ax_c_feas):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8, color=MUTED)

    predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
    panel_a(ax_a_abs, ax_a_inc, data["reports"], predictors, args.control)
    panel_b(ax_b_fid, ax_b_joint, data["certificates"], data["contrasts"])
    panel_c(ax_c_damage, ax_c_feas, data["contrasts"], data["curves"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    rendered = sum(bool(v) for v in data.values())
    print(f"wrote {out} ({rendered}/4 input groups present)")


if __name__ == "__main__":
    main()
