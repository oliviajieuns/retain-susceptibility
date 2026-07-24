"""Render the paper's two main claim tables from the validated ledger.

Every cell falls back to the explicit ``\\tblph`` placeholder when its
evidence block is incomplete; a partial campaign can therefore regenerate the
tables at any time without a favorable number ever appearing ahead of its
eligibility checks.  Claim eligibility/pass flags come exclusively from
:mod:`rsus.evidence.decisions`; this module formats, it never decides.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .registry import EvidenceContract
from .schemas import (
    PROTECTION_COMPARATORS,
    PROTECTION_OUTCOMES,
    Effect,
    EvidenceLedger,
    EvidenceRow,
    EvidenceValidationError,
)

PLACEHOLDER = r"\tblph"

# Frozen outcome-blind engineering floors for loss-shake fidelity (RQ2).
FIDELITY_TAU_RHO = 0.80
FIDELITY_TAU_K = 0.70

PARENT_LABELS = {
    "graddiff": "GradDiff",
    "npo": "NPO",
    "simnpo": "SimNPO",
    "gru": "GRU",
    "rmu": "RMU",
    "repnoise": "RepNoise",
    "circuit_breakers": "CB",
}

READOUT_HEADINGS = {
    "output_readout": r"\emph{Output-readout parents}",
    "representation_readout": r"\emph{Representation-readout parents}",
}

AXIS_BY_ROLE = {
    "primary": "Request",
    "scale_boundary": "Model",
    "model_scale": "Model",
    "model_family": "Model",
    "dataset_replication": "Dataset",
    "stress": "Dataset",
}


def _fmt(value: float | None, digits: int = 3, sign: bool = False) -> str:
    if value is None:
        return PLACEHOLDER
    pattern = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    return pattern.format(value)


def _fmt_effect(effect: Effect, *, bound: str, sign: bool = False) -> str:
    """``estimate [bound]`` with the one-sided bound the claim consumes."""
    limit = effect.lower_bound if bound == "lower" else effect.upper_bound
    if effect.estimate is None or limit is None:
        return PLACEHOLDER
    return f"{_fmt(effect.estimate, sign=sign)} [{_fmt(limit, sign=sign)}]"


def _ep(decision: Mapping[str, Any] | None) -> str:
    if decision is None:
        return PLACEHOLDER
    eligible = "y" if decision.get("eligible") else "n"
    passed = "y" if decision.get("claim_pass") else "n"
    if not decision.get("eligible"):
        passed = "--"
    return f"{eligible}/{passed}"


def _row_lookup(report: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (record["setting"], record["parent"]): record
        for record in report["rows"]
    }


def _parent_order(contract: EvidenceContract, parents: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group a setting's parents by declared readout group, keeping order."""
    grouped: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for group in contract.multi_setting.parent_groups:
        members = [parent for parent in group.parents if parent in parents]
        if members:
            grouped.append((group.group_id, members))
            seen.update(members)
    leftover = [parent for parent in parents if parent not in seen]
    if leftover:
        grouped.append(("other", leftover))
    return grouped


def _fidelity_cell(fidelity: Mapping[str, Any] | None) -> str:
    if not fidelity:
        return PLACEHOLDER
    f_rho = fidelity.get("f_rho")
    f_k = fidelity.get("f_k")
    rho_lb = fidelity.get("f_rho_lb")
    k_lb = fidelity.get("f_k_lb")
    if f_rho is None or f_k is None:
        return PLACEHOLDER
    if rho_lb is None or k_lb is None:
        return f"{f_rho:.2f}/{f_k:.2f} [{PLACEHOLDER}]"
    return (
        f"{f_rho:.2f}/{f_k:.2f} "
        f"[{rho_lb - FIDELITY_TAU_RHO:+.2f}/{k_lb - FIDELITY_TAU_K:+.2f}]"
    )


def _rq2_pass(
    row: EvidenceRow | None,
    prediction_decision: Mapping[str, Any] | None,
    fidelity: Mapping[str, Any] | None,
) -> str:
    """RQ2 = fidelity floors + added value beyond proximity and the control.

    The composition uses only bounds that were computed upstream; whenever any
    member is missing the row cannot become eligible, let alone pass.
    """
    if row is None or prediction_decision is None:
        return PLACEHOLDER
    members_present = (
        fidelity is not None
        and fidelity.get("f_rho_lb") is not None
        and fidelity.get("f_k_lb") is not None
        and row.prediction.vs_s1.complete_for_gain()
        and row.prediction.vs_control.complete_for_gain()
    )
    eligible = bool(prediction_decision.get("eligible")) and members_present
    if not eligible:
        return "n/--"
    passed = (
        fidelity["f_rho_lb"] - FIDELITY_TAU_RHO > 0.0
        and fidelity["f_k_lb"] - FIDELITY_TAU_K > 0.0
        and row.prediction.vs_s1.lower_bound is not None
        and row.prediction.vs_s1.lower_bound > 0.0
        and row.prediction.vs_control.lower_bound is not None
        and row.prediction.vs_control.lower_bound > 0.0
    )
    return f"y/{'y' if passed else 'n'}"


def _min_gain_cell(row: EvidenceRow) -> str:
    effects = (row.prediction.vs_s0, row.prediction.vs_s1)
    if not all(effect.complete_for_gain() for effect in effects):
        return PLACEHOLDER
    estimate = min(effect.estimate for effect in effects)
    bound = min(effect.lower_bound for effect in effects)
    return f"{_fmt(estimate)} [{_fmt(bound)}]"


def _tail_cell(row: EvidenceRow) -> str:
    tail = row.prediction.tail_lift
    eligible_n = row.prediction.tail_eligible_n
    total_n = row.prediction.tail_total_n
    if eligible_n is None or total_n is None:
        return PLACEHOLDER
    counts = f"{eligible_n}/{total_n}"
    if not tail.complete_for_gain():
        return f"{PLACEHOLDER}; {counts}"
    return f"{_fmt(tail.estimate, sign=True)} [{_fmt(tail.lower_bound, sign=True)}]; {counts}"


def _joint_cell(row: EvidenceRow) -> str:
    if row.prediction.joint_rho is None or row.prediction.joint.lower_bound is None:
        return PLACEHOLDER
    return f"{_fmt(row.prediction.joint_rho)} [{_fmt(row.prediction.joint.lower_bound)}]"


def _absolute_cell(row: EvidenceRow, arm: str) -> str:
    outcomes = row.protection.absolute.get(arm)
    if not outcomes or any(
        outcomes.get(outcome) is None for outcome in PROTECTION_OUTCOMES
    ):
        return PLACEHOLDER
    return f"{_fmt(outcomes['mean'])}; {_fmt(outcomes['cvar95'])}"


def _max_delta_cell(row: EvidenceRow) -> str:
    estimates: list[float] = []
    uppers: list[float] = []
    for comparator in PROTECTION_COMPARATORS:
        outcomes = row.protection.comparisons.get(comparator, {})
        for outcome in PROTECTION_OUTCOMES:
            effect = outcomes.get(outcome)
            if effect is None or not effect.complete_for_reduction():
                return PLACEHOLDER
            estimates.append(effect.estimate)
            uppers.append(effect.upper_bound)
    return f"{_fmt(max(estimates), sign=True)} [{_fmt(max(uppers), sign=True)}]"


def _min_native_cell(row: EvidenceRow) -> str:
    effects = [
        row.protection.native.get(comparator)
        for comparator in PROTECTION_COMPARATORS
    ]
    if any(effect is None or not effect.complete_for_gain() for effect in effects):
        return PLACEHOLDER
    estimate = min(effect.estimate for effect in effects)
    bound = min(effect.lower_bound for effect in effects)
    return f"{_fmt(estimate, sign=True)} [{_fmt(bound, sign=True)}]"


def _slack_cell(row: EvidenceRow) -> str:
    forget = row.protection.min_forget_margin
    utility = row.protection.min_utility_margin
    if forget is None or utility is None:
        return PLACEHOLDER
    return f"{_fmt(forget, digits=2, sign=True)}/{_fmt(utility, digits=2, sign=True)}"


def _updates_cell(row: EvidenceRow) -> str:
    diagnostics = row.protection.update_diagnostics
    if not diagnostics:
        return PLACEHOLDER
    accepted = diagnostics.get("accepted")
    rolled = diagnostics.get("rolled_back")
    if accepted is None or rolled is None:
        return PLACEHOLDER
    return f"{accepted:.0f}/{rolled:.0f}"


def render_core_evidence_table(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    report: Mapping[str, Any],
    *,
    fidelity: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Render ``tab:core-evidence`` (per-parent panels A and B)."""
    table = contract.tables.get("main_core_evidence")
    if table is None:
        raise EvidenceValidationError("contract does not register main_core_evidence")
    decisions = _row_lookup(report)
    fidelity = fidelity or {}
    lines = [
        "% Generated by experiments/paper/build_evidence.py; do not edit by hand.",
        "% Incomplete evidence remains an explicit \\tblph placeholder.",
        r"\begin{table*}[!t]",
        r"\caption{\textbf{Claim-bearing evidence by parent.}",
        r"RQ1 requires positive joint prediction, positive gains over both components,",
        r"and above-chance tail lift with at least 80\% coverage. RQ2 requires the",
        r"predeclared exact-energy fidelity floors, positive added value beyond",
        r"proximity, and a positive gain over the strongest simple control. RQ3 requires eight",
        r"$\Delta$NLL superiority bounds, four native-metric non-inferiority bounds, and",
        r"all-arm feasibility. \texttt{E/P} denotes eligible/pass.}",
        r"\label{tab:core-evidence}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
    ]
    for setting_id in table.settings:
        setting = contract.settings[setting_id]
        setting_fidelity = fidelity.get(setting_id)
        grouped = _parent_order(contract, setting.parents)

        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{@{}lccccccc@{}}")
        lines.append(r"\toprule")
        lines.append(
            r"\multicolumn{8}{@{}l}{\textbf{A. Prospective prediction and "
            rf"loss-shake validation ({setting.dataset}, {setting.model})}}}} \\"
        )
        lines.append(
            r"Parent & Joint $\rho$ [LB] & $\min(g_G,g_H)$ [min LB] &"
            r" $f_\rho/f_K$ [margin LB] & $g_{\rm ctl}$ [LB] &"
            r" $L_{\rm tail}$ [LB]; eligible $n/N$ & RQ1 E/P & RQ2 E/P \\"
        )
        lines.append(r"\midrule")
        for group_index, (group_id, members) in enumerate(grouped):
            if group_index:
                lines.append(r"\addlinespace")
            heading = READOUT_HEADINGS.get(group_id)
            if heading:
                lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{{heading}}} \\")
            for parent in members:
                key = (setting_id, parent)
                row = ledger.rows.get(key)
                decision = decisions.get(key)
                prediction_decision = decision["prediction"] if decision else None
                label = PARENT_LABELS.get(parent, parent)
                if row is None:
                    cells = [PLACEHOLDER] * 6 + [
                        _ep(prediction_decision),
                    ]
                    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
                    continue
                cells = [
                    _joint_cell(row),
                    _min_gain_cell(row),
                    _fidelity_cell(setting_fidelity),
                    _fmt_effect(row.prediction.vs_control, bound="lower"),
                    _tail_cell(row),
                    _ep(prediction_decision),
                    _rq2_pass(row, prediction_decision, setting_fidelity),
                ]
                lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")

        lines.append("")
        lines.append(r"\vspace{3pt}")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{@{}lccccccc@{}}")
        lines.append(r"\toprule")
        lines.append(
            r"\multicolumn{8}{@{}l}{\textbf{B. Constraint-matched fixed-budget "
            rf"protection ({setting.dataset}, {setting.model})}}}} \\"
        )
        lines.append(
            r"Parent & Profile mean; CVaR & No-repair mean; CVaR &"
            r" $\max_{a,k}\Delta_{a,k}$ [UCB] & $\min_a h_a$ [LB] &"
            r" min F/U slack & updates/rollback & RQ3 E/P \\"
        )
        lines.append(r"\midrule")
        for group_index, (group_id, members) in enumerate(grouped):
            if group_index:
                lines.append(r"\addlinespace")
            heading = READOUT_HEADINGS.get(group_id)
            if heading:
                lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{{heading}}} \\")
            for parent in members:
                key = (setting_id, parent)
                row = ledger.rows.get(key)
                decision = decisions.get(key)
                protection_decision = decision["protection"] if decision else None
                label = PARENT_LABELS.get(parent, parent)
                if row is None:
                    cells = [PLACEHOLDER] * 6 + [_ep(protection_decision)]
                    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
                    continue
                cells = [
                    _absolute_cell(row, "joint"),
                    _absolute_cell(row, "no_repair"),
                    _max_delta_cell(row),
                    _min_native_cell(row),
                    _slack_cell(row),
                    _updates_cell(row),
                    _ep(protection_decision),
                ]
                lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def _setting_label(setting) -> str:
    if setting.role in ("model_scale", "model_family", "scale_boundary"):
        label = setting.model
    else:
        label = setting.dataset
    if setting.role == "stress":
        label += " (stress)"
    if setting.role == "primary":
        label = f"held-out {setting.dataset} requests"
    if setting.role == "scale_boundary":
        label += " (boundary)"
    return label


FAILURE_FLAGS = (
    ("missing row", "not attempted"),
    ("not attempted", "not attempted"),
    ("incomplete planned trajectories", "incomplete"),
    ("not all predeclared profiles are valid", "invalid profile"),
    ("prediction support below frozen minimum", "non-reach"),
    ("protection support below frozen minimum", "non-reach"),
    ("prediction lacks complete common support", "no common support"),
    ("five arms do not share complete outcome support", "no common support"),
    ("not all five claim arms are feasible", "infeasible arm"),
    ("forgetting or utility constraint failed", "constraint fail"),
    ("tail lift bound or eligible coverage failed", "tail miss"),
    ("IUT failed", "IUT fail"),
    ("effects incomplete", "incomplete"),
    ("bounds incomplete", "incomplete"),
    ("weight unresolved or fallback", "fallback weight"),
)


def _failure_modes(decisions: Sequence[Mapping[str, Any]]) -> str:
    flags: list[str] = []
    for decision in decisions:
        for claim in ("prediction", "protection"):
            for reason in decision[claim].get("reasons", ()):
                for needle, flag in FAILURE_FLAGS:
                    if needle in reason:
                        if flag not in flags:
                            flags.append(flag)
                        break
    if not flags:
        return "none"
    return "; ".join(flags[:3])


def _sum_funnel(rows: Sequence[EvidenceRow | None], field: str) -> int:
    return sum(getattr(row.funnel, field) for row in rows if row is not None)


def render_robustness_table(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    report: Mapping[str, Any],
    *,
    fidelity: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Render ``tab:robustness`` (per-setting coverage and boundaries)."""
    table = contract.tables.get("main_robustness")
    if table is None:
        raise EvidenceValidationError("contract does not register main_robustness")
    decisions = _row_lookup(report)
    fidelity = fidelity or {}
    lines = [
        "% Generated by experiments/paper/build_evidence.py; do not edit by hand.",
        "% Incomplete evidence remains an explicit \\tblph placeholder.",
        r"\begin{table*}[!t]",
        r"\caption{\textbf{Predeclared coverage and failure boundaries.}",
        r"Every planned setting--parent row remains in the denominator when unfinished,",
        r"profile-invalid, gate-non-reaching, or repair-infeasible. Least-favorable",
        r"bounds are descriptive extrema; stress settings cannot rescue a primary",
        r"failure.}",
        r"\label{tab:robustness}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llccccccccl@{}}",
        r"\toprule",
        r"Axis & Setting & Plan/done & RQ1 E/P & RQ2 E/P & RQ3 E/P &"
        r" valid/reach & tail/common $n$ & all-arm feasible &"
        r" worst RQ1/RQ2/RQ3 bounds & Failure modes \\",
        r"\midrule",
    ]
    previous_axis: str | None = None
    for setting_id in table.settings:
        setting = contract.settings[setting_id]
        axis = AXIS_BY_ROLE.get(setting.role, "Setting")
        summary = report["settings"][setting_id]
        parents = setting.parents
        rows = [ledger.rows.get((setting_id, parent)) for parent in parents]
        row_decisions = [decisions[(setting_id, parent)] for parent in parents]
        planned = summary["denominators"]["planned_rows"]
        completed = summary["denominators"]["completed_rows"]

        rq1 = summary["prediction"]
        rq3 = summary["protection"]
        rq2_flags = [
            _rq2_pass(
                ledger.rows.get((setting_id, parent)),
                decisions[(setting_id, parent)]["prediction"],
                fidelity.get(setting_id),
            )
            for parent in parents
        ]
        rq2_eligible = sum(flag.startswith("y/") for flag in rq2_flags)
        rq2_passed = sum(flag == "y/y" for flag in rq2_flags)

        profiles_planned = _sum_funnel(rows, "profiles_planned")
        profiles_valid = _sum_funnel(rows, "profiles_valid")
        reached = _sum_funnel(rows, "trajectories_reached")
        trajectories_planned = _sum_funnel(rows, "trajectories_planned")
        prediction_common = _sum_funnel(rows, "prediction_common")
        feasible = _sum_funnel(rows, "protection_feasible_all_arms")
        reached_valid = _sum_funnel(rows, "reached_with_valid_profile")
        tail_eligible = sum(
            row.prediction.tail_eligible_n or 0 for row in rows if row is not None
        )

        # Least-favorable descriptive extrema per research question.  RQ1 and
        # RQ2 members are one-sided lower bounds (worst = min); the RQ3 damage
        # contrasts are one-sided upper bounds (worst = max).  The four native
        # non-inferiority lower bounds live on a different scale and stay out
        # of the single RQ3 scalar.
        rq1_bounds = [
            bound
            for row in rows
            if row is not None
            for bound in (
                row.prediction.joint.lower_bound,
                row.prediction.vs_s0.lower_bound,
                row.prediction.vs_s1.lower_bound,
                row.prediction.tail_lift.lower_bound,
            )
            if bound is not None
        ]
        rq2_bounds = [
            bound
            for row in rows
            if row is not None
            for bound in (
                row.prediction.vs_s1.lower_bound,
                row.prediction.vs_control.lower_bound,
            )
            if bound is not None
        ]
        setting_fidelity = fidelity.get(setting_id) or {}
        if rq2_bounds and setting_fidelity.get("f_rho_lb") is not None:
            rq2_bounds.append(setting_fidelity["f_rho_lb"] - FIDELITY_TAU_RHO)
        if rq2_bounds and setting_fidelity.get("f_k_lb") is not None:
            rq2_bounds.append(setting_fidelity["f_k_lb"] - FIDELITY_TAU_K)
        protection_bounds = [
            effect.upper_bound
            for row in rows
            if row is not None
            for outcomes in row.protection.comparisons.values()
            for effect in outcomes.values()
            if effect.upper_bound is not None
        ]
        worst_rq1 = min(rq1_bounds) if rq1_bounds else None
        worst_rq2 = min(rq2_bounds) if rq2_bounds else None
        worst_rq3 = max(protection_bounds) if protection_bounds else None
        if worst_rq1 is None and worst_rq2 is None and worst_rq3 is None:
            worst_cell = PLACEHOLDER
        else:
            worst_cell = " / ".join(
                _fmt(value, sign=True) if value is not None else PLACEHOLDER
                for value in (worst_rq1, worst_rq2, worst_rq3)
            )

        attempted_any = any(row is not None for row in rows)
        cells = [
            axis if axis != previous_axis else "",
            _setting_label(setting),
            f"{planned}/{completed}",
            f"{rq1['eligible']}/{rq1['passed']}",
            f"{rq2_eligible}/{rq2_passed}" if attempted_any else PLACEHOLDER,
            f"{rq3['eligible']}/{rq3['passed']}",
            (
                f"{profiles_valid}/{profiles_planned}; {reached}/{trajectories_planned}"
                if attempted_any
                else PLACEHOLDER
            ),
            (
                f"{tail_eligible}/{prediction_common}"
                if attempted_any
                else PLACEHOLDER
            ),
            f"{feasible}/{reached_valid}" if attempted_any else PLACEHOLDER,
            worst_cell,
            _failure_modes(row_decisions),
        ]
        lines.append(" & ".join(cells) + r" \\")
        previous_axis = axis
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def write_tex_tables(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    report: Mapping[str, Any],
    paper_root: str | Path,
    *,
    fidelity: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Path]:
    from .rendering import _atomic_write  # shared atomic-replace helper

    root = Path(paper_root).resolve()
    if not root.is_dir() or not (root / "main.tex").is_file():
        raise EvidenceValidationError(
            f"--paper-root must contain main.tex, got {root}"
        )
    outputs = []
    rendered = {
        contract.core_table_output: render_core_evidence_table(
            contract, ledger, report, fidelity=fidelity
        ),
        contract.robustness_table_output: render_robustness_table(
            contract, ledger, report, fidelity=fidelity
        ),
    }
    for relative, text in rendered.items():
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise EvidenceValidationError(
                "outputs.tex_tables must remain inside --paper-root"
            ) from error
        _atomic_write(target, text)
        outputs.append(target)
    return outputs
