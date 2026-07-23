"""Claim and readiness decisions for the paper evidence registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .registry import CLAIMS, EvidenceContract
from .schemas import (
    PROTECTION_COMPARATORS,
    PROTECTION_OUTCOMES,
    EvidenceLedger,
    EvidenceRow,
    EvidenceValidationError,
)
from .statistics import intersection_union_p


@dataclass(frozen=True)
class ClaimDecision:
    data_complete: bool
    eligible: bool
    statistical_pass: bool
    claim_pass: bool
    p_iut: float | None
    reasons: tuple[str, ...]


def _prediction_decision(
    row: EvidenceRow | None, *, alpha: float, minimum_support: int
) -> ClaimDecision:
    if row is None:
        return ClaimDecision(False, False, False, False, None, ("missing row",))
    effects = (row.prediction.vs_s0, row.prediction.vs_s1)
    data_complete = (
        row.prediction.paired
        and row.prediction.joint_rho is not None
        and row.prediction.top_q_recall is not None
        and all(effect.complete_for_gain() for effect in effects)
    )
    reasons: list[str] = []
    if not row.attempted:
        reasons.append("not attempted")
    if not row.completed:
        reasons.append("incomplete planned trajectories")
    if not data_complete:
        reasons.append("prediction effects/bounds incomplete")
    selection_ok = row.prediction_selection.valid and not row.prediction_selection.fallback
    if not selection_ok:
        reasons.append("prediction weight unresolved or fallback")
    funnel = row.funnel
    profiles_ok = (
        funnel.profiles_planned > 0
        and funnel.profiles_valid == funnel.profiles_planned
    )
    if not profiles_ok:
        reasons.append("not all predeclared profiles are valid")
    support_ok = funnel.reached_with_valid_profile >= minimum_support
    if not support_ok:
        reasons.append("prediction support below frozen minimum")
    common_ok = funnel.prediction_common == funnel.reached_with_valid_profile
    if not common_ok:
        reasons.append("prediction lacks complete common support")
    eligible = all(
        (
            row.attempted,
            row.completed,
            data_complete,
            selection_ok,
            profiles_ok,
            support_ok,
            common_ok,
        )
    )
    p_iut = None
    statistical_pass = False
    if data_complete:
        p_iut = intersection_union_p(
            effect.p_one_sided for effect in effects if effect.p_one_sided is not None
        )
        statistical_pass = (
            all(effect.lower_bound is not None and effect.lower_bound > 0.0 for effect in effects)
            and p_iut <= alpha
        )
        if not statistical_pass:
            reasons.append("joint-vs-S0/S1 one-sided IUT failed")
    return ClaimDecision(
        data_complete=data_complete,
        eligible=eligible,
        statistical_pass=statistical_pass,
        claim_pass=eligible and statistical_pass,
        p_iut=p_iut,
        reasons=tuple(reasons),
    )


def _protection_decision(
    row: EvidenceRow | None, *, alpha: float, minimum_support: int
) -> ClaimDecision:
    if row is None:
        return ClaimDecision(False, False, False, False, None, ("missing row",))
    effects = []
    for comparator in PROTECTION_COMPARATORS:
        outcomes = row.protection.comparisons.get(comparator, {})
        effects.extend(outcomes.get(outcome) for outcome in PROTECTION_OUTCOMES)
    data_complete = (
        row.protection.paired
        and len(effects) == len(PROTECTION_COMPARATORS) * len(PROTECTION_OUTCOMES)
        and all(effect is not None and effect.complete_for_reduction() for effect in effects)
        and row.protection.min_forget_margin is not None
        and row.protection.min_utility_margin is not None
    )
    reasons: list[str] = []
    if not row.attempted:
        reasons.append("not attempted")
    if not row.completed:
        reasons.append("incomplete planned trajectories")
    if not data_complete:
        reasons.append("four-comparator mean/CVaR effects incomplete")
    selection_ok = row.protection_selection.valid and not row.protection_selection.fallback
    if not selection_ok:
        reasons.append("protection weight unresolved or fallback")
    funnel = row.funnel
    profiles_ok = (
        funnel.profiles_planned > 0
        and funnel.profiles_valid == funnel.profiles_planned
    )
    if not profiles_ok:
        reasons.append("not all predeclared profiles are valid")
    support_ok = funnel.reached_with_valid_profile >= minimum_support
    if not support_ok:
        reasons.append("protection support below frozen minimum")
    feasible_ok = (
        funnel.protection_feasible_all_arms == funnel.reached_with_valid_profile
    )
    if not feasible_ok:
        reasons.append("not all five claim arms are feasible")
    common_ok = funnel.protection_common == funnel.protection_feasible_all_arms
    if not common_ok:
        reasons.append("five arms do not share complete outcome support")
    constraints_ok = (
        row.protection.min_forget_margin is not None
        and row.protection.min_forget_margin >= 0.0
        and row.protection.min_utility_margin is not None
        and row.protection.min_utility_margin >= 0.0
    )
    if not constraints_ok:
        reasons.append("forgetting or utility constraint failed")
    eligible = all(
        (
            row.attempted,
            row.completed,
            data_complete,
            selection_ok,
            profiles_ok,
            support_ok,
            feasible_ok,
            common_ok,
            constraints_ok,
        )
    )
    p_iut = None
    statistical_pass = False
    if data_complete:
        complete_effects = [effect for effect in effects if effect is not None]
        p_iut = intersection_union_p(
            effect.p_one_sided
            for effect in complete_effects
            if effect.p_one_sided is not None
        )
        statistical_pass = (
            all(
                effect.upper_bound is not None and effect.upper_bound < 0.0
                for effect in complete_effects
            )
            and p_iut <= alpha
        )
        if not statistical_pass:
            reasons.append("eight-way one-sided protection IUT failed")
    # exact_norm is deliberately never consulted above: it is a same-estimand
    # reference outside the four-comparator confirmatory IUT.
    return ClaimDecision(
        data_complete=data_complete,
        eligible=eligible,
        statistical_pass=statistical_pass,
        claim_pass=eligible and statistical_pass,
        p_iut=p_iut,
        reasons=tuple(reasons),
    )


def _setting_summary(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    row_decisions: Mapping[tuple[str, str], Mapping[str, ClaimDecision]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for setting_id, setting in contract.settings.items():
        setting_rows = [ledger.rows.get((setting_id, parent)) for parent in setting.parents]
        by_claim: dict[str, object] = {
            "denominators": {
                "planned_rows": len(setting.parents),
                "attempted_rows": sum(bool(row and row.attempted) for row in setting_rows),
                "completed_rows": sum(bool(row and row.completed) for row in setting_rows),
            }
        }
        for claim in CLAIMS:
            decisions = [row_decisions[(setting_id, parent)][claim] for parent in setting.parents]
            by_claim[claim] = {
                "planned": len(decisions),
                "data_complete": sum(item.data_complete for item in decisions),
                "eligible": sum(item.eligible for item in decisions),
                "passed": sum(item.claim_pass for item in decisions),
                "pass": bool(decisions) and all(item.claim_pass for item in decisions),
            }
        parent_groups = []
        for group in contract.multi_setting.parent_groups:
            corrected_alpha = contract.alpha / len(group.parents)
            passed = []
            for parent in group.parents:
                decisions = [
                    row_decisions[(setting_id, parent)][claim]
                    for claim in CLAIMS
                ]
                if all(
                    decision.claim_pass
                    and decision.p_iut is not None
                    and decision.p_iut <= corrected_alpha
                    for decision in decisions
                ):
                    passed.append(parent)
            parent_groups.append({
                "id": group.group_id,
                "minimum_joint_pass": group.minimum_joint_pass,
                "multiplicity": group.multiplicity,
                "familywise_alpha": contract.alpha,
                "per_parent_iut_alpha": corrected_alpha,
                "passed_parents": passed,
                "pass_count": len(passed),
                "planned_count": len(group.parents),
                "pass": len(passed) >= group.minimum_joint_pass,
            })
        by_claim["chain"] = {
            "parent_groups": parent_groups,
            "pass": all(group["pass"] for group in parent_groups),
        }
        result[setting_id] = by_claim
    return result


def _multi_setting_summary(
    contract: EvidenceContract, setting_summary: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    rule = contract.multi_setting
    primary_pass = {
        setting: bool(setting_summary[setting]["chain"]["pass"])
        for setting in rule.primary_required
    }
    group_results = []
    for group in rule.groups:
        passes = [
            setting
            for setting in group.settings
            if bool(setting_summary[setting]["chain"]["pass"])
        ]
        group_results.append({
            "id": group.group_id,
            "minimum_pass": group.minimum_pass,
            "passed_settings": passes,
            "pass_count": len(passes),
            "planned_count": len(group.settings),
            "pass": len(passes) >= group.minimum_pass,
        })
    joint = all(primary_pass.values()) and all(group["pass"] for group in group_results)
    return {
        "rule_id": rule.rule_id,
        "stress_excluded": list(rule.stress_excluded),
        "primary": primary_pass,
        "groups": group_results,
        "setting_support": (
            "minimum joint prediction+protection passes per readout group after "
            "within-group Bonferroni correction"
        ),
        "pass": joint,
    }


def evaluate_evidence(
    contract: EvidenceContract, ledger: EvidenceLedger
) -> dict[str, object]:
    """Evaluate every predeclared row, including rows absent from the ledger."""
    planned = set(contract.planned_keys)
    extra = set(ledger.rows) - planned
    if extra:
        raise EvidenceValidationError(
            f"ledger contains unregistered setting/parent rows: {sorted(extra)}"
        )
    unknown_artifacts = set(ledger.artifacts) - set(contract.artifacts)
    if unknown_artifacts:
        raise EvidenceValidationError(
            f"ledger contains unregistered artifacts: {sorted(unknown_artifacts)}"
        )

    row_decisions: dict[tuple[str, str], dict[str, ClaimDecision]] = {}
    rows_json: list[dict[str, object]] = []
    for key in contract.planned_keys:
        row = ledger.rows.get(key)
        prediction = _prediction_decision(
            row,
            alpha=contract.alpha,
            minimum_support=contract.minimum_support_units,
        )
        protection = _protection_decision(
            row,
            alpha=contract.alpha,
            minimum_support=contract.minimum_support_units,
        )
        row_decisions[key] = {
            "prediction": prediction,
            "protection": protection,
        }
        rows_json.append(
            {
                "setting": key[0],
                "parent": key[1],
                "attempted": bool(row and row.attempted),
                "completed": bool(row and row.completed),
                "funnel": asdict(row.funnel) if row else None,
                "prediction_alpha": (
                    row.prediction_selection.alpha if row else None
                ),
                "protection_alpha": (
                    row.protection_selection.alpha if row else None
                ),
                "prediction": asdict(prediction),
                "protection": asdict(protection),
            }
        )

    settings = _setting_summary(contract, ledger, row_decisions)
    multi = _multi_setting_summary(contract, settings)
    tables: dict[str, object] = {}
    for table_id, table in contract.tables.items():
        selected_keys = [
            (setting, parent)
            for setting in table.settings
            for parent in contract.settings[setting].parents
        ]
        incomplete_rows = []
        for key in selected_keys:
            row = ledger.rows.get(key)
            if row is None or not row.completed:
                incomplete_rows.append(f"{key[0]}::{key[1]}")
                continue
            if any(not row_decisions[key][claim].data_complete for claim in table.claims):
                incomplete_rows.append(f"{key[0]}::{key[1]}")
        missing_artifacts = [
            artifact
            for artifact in table.artifacts
            if not ledger.artifacts.get(artifact)
            or not ledger.artifacts[artifact].completed
        ]
        tables[table_id] = {
            "label": table.label,
            "location": table.location,
            "producer": table.producer,
            "planned_rows": len(selected_keys),
            "completed_rows": len(selected_keys) - len(incomplete_rows),
            "incomplete_rows": incomplete_rows,
            "missing_artifacts": missing_artifacts,
            "ready": not incomplete_rows and not missing_artifacts,
        }

    attempted = sum(bool(ledger.rows.get(key) and ledger.rows[key].attempted) for key in planned)
    completed = sum(bool(ledger.rows.get(key) and ledger.rows[key].completed) for key in planned)
    return {
        "schema_version": 1,
        "decision_alpha": contract.alpha,
        "minimum_support_units": contract.minimum_support_units,
        "denominators": {
            "planned_rows": len(planned),
            "attempted_rows": attempted,
            "completed_rows": completed,
            "missing_rows": len(planned - set(ledger.rows)),
        },
        "rows": rows_json,
        "settings": settings,
        "multi_setting": multi,
        "tables": tables,
        "all_tables_ready": all(bool(table["ready"]) for table in tables.values()),
    }
