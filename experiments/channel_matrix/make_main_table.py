"""Render the compact, uncertainty-bearing main channel matrix from aggregates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OBJ_TEX = {
    "ga": "GA",
    "graddiff": "GradDiff",
    "npo": "NPO+GD",
    "simnpo": "SimNPO+GD",
    "gru": "GRU",
    "rmu": "RMU",
    "repnoise": "RepNoise",
    "circuit_breakers": "RR/CB",
    "idkdpo": "IdkDPO",
}
ADAPTED_OBJECTIVES = {"repnoise", "circuit_breakers"}
PRED_TEX = {
    "grad_norm": "Exact gradient norm",
    "fd_norm": r"Randomized sensitivity (ours)",
    "knn_feature": "Hidden-state proximity",
    "knn_embed": "Sentence-embedding proximity",
    "knn_lexical": "Lexical overlap",
}
PRED_ORDER = ["grad_norm", "fd_norm", "knn_feature", "knn_embed", "knn_lexical"]
BWD_FREE = {
    "grad_norm": "no",
    "fd_norm": "yes",
    "knn_feature": "yes",
    "knn_embed": "yes",
    "knn_lexical": "yes",
}


def _fmt(point: float, lo: float, hi: float, bold: bool) -> str:
    body = rf"\shortstack{{{point:.2f}\\{{\tiny [{lo:.2f},{hi:.2f}]}}}}"
    return rf"\textbf{{{body}}}" if bold else body


def _objective_label(objective: str, summary: dict) -> str:
    markers = []
    if objective in ADAPTED_OBJECTIVES:
        markers.append("*")
    status = summary["objective_status"][objective]
    if status["failed_runs"]:
        markers.append(r"\dagger")
    if status["collapsed_runs"]:
        markers.append(r"\ddagger")
    suffix = rf"$^{{{''.join(markers)}}}$" if markers else ""
    return OBJ_TEX.get(objective, objective) + suffix


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stress-out", default="",
                   help="optional appendix table for predeclared stress objectives")
    a = p.parse_args()

    with open(a.report, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    objectives = summary["objectives"]
    output = [o for o in objectives if next(r for r in rows if r["objective"] == o)["channel"] == "loss_gradient"]
    representation = [o for o in objectives if o not in output]
    predictors = [p_ for p_ in PRED_ORDER if any(r["predictor"] == p_ for r in rows)]
    cell = {(r["predictor"], r["objective"]): r for r in rows}
    best = {
        objective: max(float(cell[predictor, objective]["rho"]) for predictor in predictors)
        for objective in objectives
    }
    inter = summary["roster_interaction"]
    protocol = summary["protocol"]
    candidate_counts = protocol["audit_candidates_per_run"]
    candidate_text = str(candidate_counts[0]) if len(candidate_counts) == 1 else str(candidate_counts)

    columns = "@{}llc" + "c" * len(objectives) + "@{}"
    lines = [
        r"\begin{table*}[t]",
        (r"\caption{\textbf{Channel-conditioned prediction of realized damage at 7B/8B scale.} "
         f"Pooled within-run Spearman $\\rho$ across {summary['n_runs']} sealed runs, "
         f"{len(summary['models'])} model(s), {len(summary['requests'])} request(s), and "
         f"{len(summary['seeds'])} seed(s), with $n={candidate_text}$ audit candidates per run. "
         f"All parents update the declared last-{protocol['probe_config']['block_last_n']} "
         f"MLP-down block in {protocol['dtype']}; brackets are hierarchical-bootstrap 95\\% CIs. "
         f"The predeclared extension endpoint is $\\Delta={inter['delta']:+.3f}$ "
         f"$[{inter['lo']:+.3f},{inter['hi']:+.3f}]$. "
         r"$^{*}$Controlled representation-loss adaptations; implementation differences from "
         r"the original safety methods are disclosed in the appendix. "
         r"$^{\dagger}$At least one run missed the frozen forget criterion; "
         r"$^{\ddagger}$at least one run crossed the frozen collapse threshold.}" ),
        r"\label{tab:channel-matrix-7b}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        (rf"Family & Predictor & Bwd-free & \multicolumn{{{len(output)}}}{{c}}{{\textbf{{Output / loss-gradient}}}} "
         rf"& \multicolumn{{{len(representation)}}}{{c}}{{\textbf{{Representation}}}} \\"),
        (rf"\cmidrule(lr){{4-{3 + len(output)}}}"
         rf"\cmidrule(lr){{{4 + len(output)}-{3 + len(objectives)}}}"),
        " &  &  & " + " & ".join(_objective_label(o, summary) for o in objectives) + r" \\",
        r"\midrule",
    ]
    previous_family = None
    for predictor in predictors:
        family = "Gradient" if predictor in {"grad_norm", "fd_norm"} else "Representation"
        if previous_family is not None and family != previous_family:
            lines.append(r"\addlinespace")
        family_cell = family if family != previous_family else ""
        previous_family = family
        values = []
        for objective in objectives:
            row = cell[predictor, objective]
            point = float(row["rho"])
            values.append(_fmt(
                point,
                float(row["rho_lo"]),
                float(row["rho_hi"]),
                point == best[objective],
            ))
        lines.append(
            f"{family_cell} & {PRED_TEX[predictor]} & {BWD_FREE[predictor]} & "
            + " & ".join(values) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")

    if a.stress_out and summary.get("stress_objectives"):
        stress = summary["stress_objectives"]
        stress_best = {
            objective: max(float(cell[predictor, objective]["rho"]) for predictor in predictors)
            for objective in stress
        }
        stress_lines = [
            r"\begin{table}[t]",
            (r"\caption{\textbf{Predeclared output-channel stress controls.} "
             r"These trajectories completed before susceptibility scores were opened but are "
             r"excluded from the core interaction. Brackets are hierarchical-bootstrap 95\% "
             r"CIs. $^{\dagger}$At least one run missed the frozen forget criterion; "
             r"$^{\ddagger}$at least one run crossed the frozen collapse threshold.}"),
            r"\label{tab:channel-stress-7b}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\begin{tabular}{@{}lcc" + "c" * len(stress) + r"@{}}",
            r"\toprule",
            "Family & Predictor & Bwd-free & "
            + " & ".join(_objective_label(objective, summary) for objective in stress)
            + r" \\",
            r"\midrule",
        ]
        previous_family = None
        for predictor in predictors:
            family = "Gradient" if predictor in {"grad_norm", "fd_norm"} else "Representation"
            if previous_family is not None and family != previous_family:
                stress_lines.append(r"\addlinespace")
            family_cell = family if family != previous_family else ""
            previous_family = family
            values = []
            for objective in stress:
                row = cell[predictor, objective]
                point = float(row["rho"])
                values.append(_fmt(
                    point,
                    float(row["rho_lo"]),
                    float(row["rho_hi"]),
                    point == stress_best[objective],
                ))
            stress_lines.append(
                f"{family_cell} & {PRED_TEX[predictor]} & {BWD_FREE[predictor]} & "
                + " & ".join(values) + r" \\"
            )
        stress_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        Path(a.stress_out).write_text("\n".join(stress_lines) + "\n", encoding="utf-8")
        print(f"wrote {a.stress_out}")


if __name__ == "__main__":
    main()
