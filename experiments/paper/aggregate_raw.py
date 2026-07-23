"""Build the normalized paper ledger from candidate-level JSONL shards.

Example
-------
python experiments/paper/aggregate_raw.py \
  --plan results/paper/raw_plan.json \
  --prediction-raw runs/tofu/prediction.jsonl \
  --protection-raw runs/tofu/protection.jsonl \
  --out results/paper/evidence_ledger.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.evidence.raw import (  # noqa: E402
    aggregate_raw_evidence,
    build_raw_artifacts,
    load_raw_plan,
    read_raw_records,
    write_ledger,
)
from rsus.evidence.schemas import EvidenceLedger, EvidenceValidationError  # noqa: E402


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _artifact_mapping(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    source = _path(path)
    ledger = EvidenceLedger.read(source)
    # Re-read the JSON because the aggregate schema intentionally exposes no
    # serialization API for arbitrary artifact metadata/headline text.
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(
            f"cannot preserve artifacts from {source}: {error}"
        ) from error
    artifacts = raw.get("artifacts", {}) or {}
    if not isinstance(artifacts, Mapping):
        raise EvidenceValidationError("artifacts-from ledger has invalid artifacts")
    # The first read above validates every ArtifactStatus field.
    assert ledger.artifacts.keys() == artifacts.keys()
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="immutable raw unit plan JSON")
    parser.add_argument(
        "--prediction-raw",
        action="append",
        default=[],
        metavar="PATH",
        help="prediction JSONL/JSON shard; repeat for multiple datasets",
    )
    parser.add_argument(
        "--protection-raw",
        action="append",
        default=[],
        metavar="PATH",
        help="protection JSONL/JSON shard; repeat for multiple datasets",
    )
    parser.add_argument("--out", required=True, help="normalized ledger JSON")
    parser.add_argument(
        "--artifact-raw",
        action="append",
        default=[],
        metavar="ARTIFACT=PATH",
        help=(
            "contracted measurement shard for lse_fidelity_cost, "
            "protection_budget_sweep, or specificity_negative_controls; repeatable"
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="artifact JSON output directory; defaults beside --out",
    )
    parser.add_argument(
        "--artifacts-from",
        default=None,
        help="optional existing ledger whose validated artifacts mapping is preserved",
    )
    return parser


def _measurement_inputs(values: list[str]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for value in values:
        if "=" not in value:
            raise EvidenceValidationError(
                "--artifact-raw must use ARTIFACT=PATH syntax"
            )
        artifact_id, raw_path = value.split("=", 1)
        artifact_id = artifact_id.strip()
        if not artifact_id or not raw_path.strip():
            raise EvidenceValidationError(
                "--artifact-raw must use non-empty ARTIFACT=PATH"
            )
        result.setdefault(artifact_id, []).extend(
            read_raw_records([_path(raw_path.strip())])
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = load_raw_plan(_path(args.plan))
        prediction = read_raw_records(_path(path) for path in args.prediction_raw)
        protection = read_raw_records(_path(path) for path in args.protection_raw)
        artifacts = dict(_artifact_mapping(args.artifacts_from))
        measurements = _measurement_inputs(args.artifact_raw)
        if plan.artifact_contracts:
            artifact_dir = (
                _path(args.artifact_dir)
                if args.artifact_dir
                else _path(args.out).parent / "artifacts"
            )
            produced, artifact_paths = build_raw_artifacts(
                plan,
                prediction,
                measurements,
                output_dir=artifact_dir,
            )
            artifacts.update(produced)
            for artifact_id, artifact_path in sorted(artifact_paths.items()):
                print(f"wrote artifact {artifact_id}: {artifact_path}")
        elif measurements:
            raise EvidenceValidationError(
                "--artifact-raw supplied but raw plan has no artifact_contracts"
            )
        ledger = aggregate_raw_evidence(
            plan,
            prediction,
            protection,
            artifacts=artifacts,
        )
        output = write_ledger(ledger, _path(args.out))
        attempted = sum(bool(row["attempted"]) for row in ledger["rows"])
        completed = sum(bool(row["completed"]) for row in ledger["rows"])
        print(f"plan sha256: {plan.source_sha256}")
        print(f"wrote ledger: {output}")
        print(
            "rows planned/attempted/completed: "
            f"{len(ledger['rows'])}/{attempted}/{completed}"
        )
        return 0
    except (EvidenceValidationError, ValueError) as error:
        print(f"raw evidence aggregation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
