"""Render machine-readable readiness and the paper's five headline macros."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from .registry import EvidenceContract
from .schemas import EvidenceLedger, EvidenceValidationError


PLACEHOLDERS = {
    "TailHeadline": (
        "Report tail concentration and semantic coherence with support, "
        "followed by the strongest observed boundary."
    ),
    "PredictionHeadline": (
        "Report eligible/pass parents, least-favorable joint-over-endpoint "
        "bound, and the strongest reversal."
    ),
    "FidelityHeadline": (
        "Report the frozen $(R,\\eta)$ operating point, exact-energy "
        "agreement, measured time/memory, and validity coverage."
    ),
    "ProtectionHeadline": (
        "Report eligible/pass parents, the largest of eight comparator UCBs, "
        "constraint slack, and infeasible cases."
    ),
    "TransferHeadline": (
        "Report planned/completed/eligible/pass settings, least-favorable "
        "bounds, and non-supporting settings."
    ),
}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_readiness_json(report: Mapping[str, object], path: str | Path) -> Path:
    target = Path(path)
    _atomic_write(target, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return target


def _effect_extrema(
    ledger: EvidenceLedger,
    *,
    primary_settings: set[str],
    claim: str,
) -> list[float]:
    values: list[float] = []
    for (setting, _), row in ledger.rows.items():
        if setting not in primary_settings:
            continue
        if claim == "prediction":
            for effect in (row.prediction.vs_s0, row.prediction.vs_s1):
                if effect.lower_bound is not None:
                    values.append(effect.lower_bound)
        elif claim == "protection":
            for outcomes in row.protection.comparisons.values():
                for effect in outcomes.values():
                    if effect.upper_bound is not None:
                        values.append(effect.upper_bound)
        else:  # pragma: no cover - internal caller is fixed.
            raise ValueError(claim)
    return values


def _primary_claim_headline(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    report: Mapping[str, object],
    *,
    claim: str,
) -> str | None:
    primary = {
        setting_id
        for setting_id, setting in contract.settings.items()
        if setting.role == "primary"
    }
    row_records = [
        row
        for row in report["rows"]
        if row["setting"] in primary
    ]
    if not row_records or any(
        not row["completed"] or not row[claim]["data_complete"]
        for row in row_records
    ):
        return None
    eligible = sum(row[claim]["eligible"] for row in row_records)
    passed = sum(row[claim]["claim_pass"] for row in row_records)
    total = len(row_records)
    extrema = _effect_extrema(
        ledger, primary_settings=primary, claim=claim
    )
    if claim == "prediction":
        if not extrema:
            return None
        return (
            f"Prediction passes {passed}/{total} predeclared primary parent rows "
            f"({eligible}/{total} eligible); the least-favorable endpoint-gain "
            f"lower bound is {min(extrema):.3f}."
        )
    if not extrema:
        return None
    return (
        f"Protection passes {passed}/{total} predeclared primary parent rows "
        f"({eligible}/{total} eligible); the largest of the eight comparator "
        f"upper bounds is {max(extrema):.3f}."
    )


def _artifact_headline(ledger: EvidenceLedger, artifact: str) -> str | None:
    status = ledger.artifacts.get(artifact)
    if status and status.completed and status.headline_tex:
        return status.headline_tex.strip()
    return None


def _transfer_headline(
    contract: EvidenceContract, report: Mapping[str, object]
) -> str | None:
    table = report["tables"].get("main_robustness")
    if not table or not table["ready"]:
        return None
    multi = report["multi_setting"]
    settings = report["settings"]
    stress = set(contract.multi_setting.stress_excluded)
    planned = len(settings) - len(stress)
    completed = sum(
        result["denominators"]["completed_rows"]
        == result["denominators"]["planned_rows"]
        for setting, result in settings.items()
        if setting not in stress
    )
    chain = sum(
        bool(result["chain"]["pass"])
        for setting, result in settings.items()
        if setting not in stress
    )
    status = "licensed" if multi["pass"] else "not licensed"
    return (
        f"The predeclared multi-setting statement is {status}: {completed}/{planned} "
        f"non-stress settings are complete and {chain}/{planned} support the full "
        f"prediction--protection chain under the two-readout rule; stress settings "
        f"cannot rescue the rule."
    )


def render_tex_macros(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    report: Mapping[str, object],
) -> str:
    """Render exact macro names consumed by the authoritative paper.

    A relevant result is emitted only after its whole evidence block is
    complete.  Otherwise the existing explicit result placeholder is kept;
    partial estimates never become a prose claim.
    """
    values = {
        "TailHeadline": _artifact_headline(ledger, "tail_structure"),
        "PredictionHeadline": _primary_claim_headline(
            contract, ledger, report, claim="prediction"
        ),
        "FidelityHeadline": _artifact_headline(ledger, "lse_fidelity_cost"),
        "ProtectionHeadline": _primary_claim_headline(
            contract, ledger, report, claim="protection"
        ),
        "TransferHeadline": _transfer_headline(contract, report),
    }
    lines = [
        "% Generated by experiments/paper/build_evidence.py; do not edit by hand.",
        "% Incomplete evidence remains an explicit placeholder and cannot license a claim.",
    ]
    for macro in (
        "TailHeadline",
        "PredictionHeadline",
        "FidelityHeadline",
        "ProtectionHeadline",
        "TransferHeadline",
    ):
        value = values[macro]
        body = value if value is not None else rf"\resph{{{PLACEHOLDERS[macro]}}}"
        lines.append(rf"\renewcommand{{\{macro}}}{{{body}}}")
    return "\n".join(lines) + "\n"


def write_tex_macros(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    report: Mapping[str, object],
    paper_root: str | Path,
) -> Path:
    root = Path(paper_root).resolve()
    if not root.is_dir() or not (root / "main.tex").is_file():
        raise EvidenceValidationError(
            f"--paper-root must contain main.tex, got {root}"
        )
    target = (root / contract.tex_output).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise EvidenceValidationError(
            "outputs.tex_macros must remain inside --paper-root"
        ) from error
    # All validation and rendering happen before the atomic replacement.
    rendered = render_tex_macros(contract, ledger, report)
    _atomic_write(target, rendered)
    return target
