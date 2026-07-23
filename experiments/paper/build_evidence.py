"""Validate the complete paper evidence ledger and render claim-safe outputs.

Examples
--------
Readiness only (missing ledger rows remain in the planned denominator)::

    python experiments/paper/build_evidence.py

Validate all artifacts and atomically update the authoritative paper macros::

    python experiments/paper/build_evidence.py --paper-root ../paper --require-ready
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.evidence.decisions import evaluate_evidence  # noqa: E402
from rsus.evidence.registry import load_contract  # noqa: E402
from rsus.evidence.rendering import (  # noqa: E402
    write_readiness_json,
    write_tex_macros,
)
from rsus.evidence.tables import write_tex_tables  # noqa: E402
from rsus.evidence.schemas import (  # noqa: E402
    EvidenceLedger,
    EvidenceValidationError,
    validate_artifact_files,
)


def _load_fidelity_inputs(contract) -> dict[str, dict]:
    """Load per-setting fidelity summaries named by the frozen contract.

    A missing or malformed file keeps its setting's fidelity cells as
    placeholders rather than failing the whole render; the RQ2 composition in
    the table module cannot pass without the bounds anyway.
    """
    import json

    result: dict[str, dict] = {}
    for setting_id, relative in contract.fidelity_inputs.items():
        path = _resolve_repo_path(relative)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceValidationError(
                f"fidelity_inputs.{setting_id} is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise EvidenceValidationError(
                f"fidelity_inputs.{setting_id} root must be a mapping"
            )
        if payload.get("setting") != setting_id:
            raise EvidenceValidationError(
                f"fidelity_inputs.{setting_id} carries setting "
                f"{payload.get('setting')!r}; refusing a mismatched summary"
            )
        if payload.get("certificate_passed") is not True:
            # A failed or unverified certificate cannot fill fidelity cells.
            continue
        result[setting_id] = payload
    return result


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "paper" / "evidence.yaml"),
        help="predeclared evidence registry",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="normalized evidence ledger JSON; defaults to config.ledger",
    )
    parser.add_argument(
        "--readiness-out",
        default=None,
        help="readiness JSON path; defaults to config.outputs.readiness_json",
    )
    parser.add_argument(
        "--paper-root",
        default=None,
        help="authoritative paper root containing main.tex; writes five macros",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="return exit status 2 unless every registered table is data-ready",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = _resolve_repo_path(args.config).resolve()
        contract = load_contract(config_path)
        ledger_path = _resolve_repo_path(args.ledger or contract.ledger_path).resolve()
        ledger = EvidenceLedger.read(ledger_path) if ledger_path.is_file() else EvidenceLedger.empty()
        validate_artifact_files(ledger, repository_root=ROOT)
        report = evaluate_evidence(contract, ledger)
        report["sources"] = {
            "config": str(config_path),
            "ledger": str(ledger_path),
            "ledger_exists": ledger_path.is_file(),
        }
        readiness_path = _resolve_repo_path(
            args.readiness_out or contract.readiness_output
        ).resolve()
        write_readiness_json(report, readiness_path)
        print(f"wrote readiness: {readiness_path}")
        if args.paper_root:
            paper_root = _resolve_repo_path(args.paper_root)
            macro_path = write_tex_macros(contract, ledger, report, paper_root)
            print(f"wrote paper macros: {macro_path}")
            table_paths = write_tex_tables(
                contract,
                ledger,
                report,
                paper_root,
                fidelity=_load_fidelity_inputs(contract),
            )
            for table_path in table_paths:
                print(f"wrote paper table: {table_path}")
        denominators = report["denominators"]
        print(
            "rows planned/attempted/completed: "
            f"{denominators['planned_rows']}/"
            f"{denominators['attempted_rows']}/"
            f"{denominators['completed_rows']}"
        )
        print(
            "multi-setting claim: "
            + ("PASS" if report["multi_setting"]["pass"] else "NOT LICENSED")
        )
        if args.require_ready and not report["all_tables_ready"]:
            return 2
        return 0
    except EvidenceValidationError as error:
        print(f"evidence validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
