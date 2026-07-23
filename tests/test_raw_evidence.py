"""CPU tests for candidate-level raw evidence aggregation."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from rsus.evidence.raw import (
    aggregate_raw_evidence,
    build_raw_artifacts,
    raw_plan_from_mapping,
)
from rsus.evidence.schemas import (
    EvidenceLedger,
    EvidenceValidationError,
    validate_artifact_files,
)


ROOT = Path(__file__).resolve().parents[1]


def _selection(alpha: float = 0.5) -> dict:
    return {"valid": True, "fallback": False, "alpha": alpha}


def _plan(*, units: list[tuple[str, str]] | None = None, replicates: int = 39):
    request_seeds = units or [("r1", "1"), ("r2", "1")]
    return raw_plan_from_mapping(
        {
            "schema_version": 1,
            "bootstrap": {
                "replicates": replicates,
                "seed": 19,
                "alpha": 0.05,
                "top_q": 0.25,
                "cvar_q": 0.75,
            },
            "units": [
                {
                    "setting": "primary",
                    "parent": "npo",
                    "request": request,
                    "seed": seed,
                    "prediction_selection": _selection(0.4),
                    "protection_selection": _selection(0.6),
                    "repeated_random_draws": ["d0", "d1"],
                }
                for request, seed in request_seeds
            ],
        }
    )


def _prediction(request: str, seed: str = "1") -> list[dict]:
    # Both groups retain within-group rank variation, so semantic-group
    # bootstrap draws remain well-defined even when one group repeats.
    damage = [0.1, 0.4, 0.8, 0.2, 0.5, 0.9]
    result = []
    for index, value in enumerate(damage):
        result.append(
            {
                "setting": "primary",
                "parent": "npo",
                "request": request,
                "seed": seed,
                "candidate_id": f"c{index}",
                "group": "g0" if index < 3 else "g1",
                "s0": -value,
                "s1": float(index % 3),
                "joint": value,
                "damage": value,
                "profile_valid": True,
                "reached": True,
                "trajectory_completed": True,
                "prediction_selection": _selection(0.4),
            }
        )
    return result


def _protection(request: str, seed: str = "1") -> list[dict]:
    base = [0.2, 0.4, 0.6, 0.3, 0.5, 0.7]
    offsets = {
        "joint": 0.0,
        "no_repair": 0.5,
        "s0": 0.4,
        "s1": 0.3,
    }
    result: list[dict] = []
    for arm, offset in offsets.items():
        for index, value in enumerate(base):
            result.append(
                {
                    "setting": "primary",
                    "parent": "npo",
                    "request": request,
                    "seed": seed,
                    "candidate_id": f"c{index}",
                    "group": "g0" if index < 3 else "g1",
                    "arm": arm,
                    "damage": value + offset,
                    "feasible": True,
                    "direct_forget_margin": 0.2,
                    "paraphrase_forget_margin": 0.3,
                    "extraction_generation_margin": 0.25,
                    "utility_margin": 0.1,
                    "parent_checkpoint_id": f"checkpoint-{request}-{seed}",
                    "parent_checkpoint_first_reaching": True,
                    "protection_selection": _selection(0.6),
                }
            )
    for draw, offset in (("d0", 0.2), ("d1", 0.4)):
        for index, value in enumerate(base):
            result.append(
                {
                    "setting": "primary",
                    "parent": "npo",
                    "request": request,
                    "seed": seed,
                    "candidate_id": f"c{index}",
                    "group": "g0" if index < 3 else "g1",
                    "arm": "repeated_random",
                    "draw_id": draw,
                    "draw_complete": True,
                    "damage": value + offset,
                    "feasible": True,
                    "direct_forget_margin": 0.2,
                    "paraphrase_forget_margin": 0.3,
                    "extraction_generation_margin": 0.25,
                    "utility_margin": 0.1,
                    "parent_checkpoint_id": f"checkpoint-{request}-{seed}",
                    "parent_checkpoint_first_reaching": True,
                    "protection_selection": _selection(0.6),
                }
            )
    return result


def test_full_raw_campaign_produces_schema_valid_paired_ledger():
    plan = _plan()
    predictions = _prediction("r1") + _prediction("r2")
    protections = _protection("r1") + _protection("r2")
    raw = aggregate_raw_evidence(plan, predictions, protections)
    ledger = EvidenceLedger.from_mapping(raw)
    row = ledger.rows[("primary", "npo")]

    assert row.attempted and row.completed
    assert row.funnel.trajectories_planned == 2
    assert row.funnel.prediction_common == 2
    assert row.funnel.protection_feasible_all_arms == 2
    assert row.funnel.protection_common == 2
    assert row.prediction.paired
    assert row.prediction.joint_rho == pytest.approx(1.0)
    assert row.prediction.vs_s0.estimate == pytest.approx(2.0)
    assert row.prediction.vs_s0.lower_bound > 0
    assert row.protection.paired
    assert row.protection.comparisons["no_repair"]["mean"].estimate == pytest.approx(-0.5)
    assert row.protection.comparisons["repeated_random"]["mean"].estimate == pytest.approx(-0.3)
    assert row.protection.comparisons["s1"]["cvar95"].upper_bound < 0
    assert row.protection.min_forget_margin == pytest.approx(0.2)
    assert row.protection.min_utility_margin == pytest.approx(0.1)


def test_missing_planned_unit_stays_in_funnel_and_row_is_incomplete():
    plan = _plan()
    raw = aggregate_raw_evidence(
        plan,
        _prediction("r1"),
        _protection("r1"),
    )
    row = EvidenceLedger.from_mapping(raw).rows[("primary", "npo")]
    assert row.attempted
    assert not row.completed
    assert row.funnel.trajectories_planned == 2
    assert row.funnel.trajectories_attempted == 1
    assert row.funnel.trajectories_completed == 1
    assert row.funnel.prediction_common == 1
    assert row.funnel.protection_common == 1


def test_missing_arm_is_not_silently_intersected():
    plan = _plan(units=[("r1", "1")])
    protection = [row for row in _protection("r1") if row["arm"] != "s1"]
    raw = aggregate_raw_evidence(plan, _prediction("r1"), protection)
    row = EvidenceLedger.from_mapping(raw).rows[("primary", "npo")]
    assert row.funnel.reached_with_valid_profile == 1
    assert row.funnel.protection_feasible_all_arms == 0
    assert row.funnel.protection_common == 0
    assert not row.protection.paired
    assert row.protection.comparisons == {}


def test_feasibility_and_common_support_are_separate_funnels():
    plan = _plan(units=[("r1", "1")])
    protection = _protection("r1")
    changed = next(
        row
        for row in protection
        if row["arm"] == "s1" and row["candidate_id"] == "c5"
    )
    changed["candidate_id"] = "different-support-id"
    raw = aggregate_raw_evidence(plan, _prediction("r1"), protection)
    row = EvidenceLedger.from_mapping(raw).rows[("primary", "npo")]
    assert row.funnel.protection_feasible_all_arms == 1
    assert row.funnel.protection_common == 0
    assert not row.protection.paired


def test_extraction_generation_margin_is_mandatory_and_fail_closed():
    plan = _plan(units=[("r1", "1")])
    protection = _protection("r1")
    protection[0].pop("extraction_generation_margin")
    with pytest.raises(EvidenceValidationError, match="extraction_generation_margin"):
        aggregate_raw_evidence(plan, _prediction("r1"), protection)

    inconsistent = _protection("r1")
    inconsistent[0]["extraction_generation_margin"] = -0.01
    with pytest.raises(EvidenceValidationError, match="must equal the conjunction"):
        aggregate_raw_evidence(plan, _prediction("r1"), inconsistent)


def test_every_arm_must_share_the_first_reaching_parent_checkpoint():
    plan = _plan(units=[("r1", "1")])
    changed = _protection("r1")
    changed[-1]["parent_checkpoint_id"] = "different-checkpoint"
    with pytest.raises(EvidenceValidationError, match="mixes parent checkpoints"):
        aggregate_raw_evidence(plan, _prediction("r1"), changed)

    not_first = _protection("r1")
    not_first[0]["parent_checkpoint_first_reaching"] = False
    with pytest.raises(EvidenceValidationError, match="first direct-criterion-reaching"):
        aggregate_raw_evidence(plan, _prediction("r1"), not_first)


def test_incomplete_repeated_random_draw_fails_all_arm_funnel():
    plan = _plan(units=[("r1", "1")])
    protection = _protection("r1")
    next(
        row
        for row in protection
        if row["arm"] == "repeated_random" and row["draw_id"] == "d1"
    )["draw_complete"] = False
    raw = aggregate_raw_evidence(plan, _prediction("r1"), protection)
    row = EvidenceLedger.from_mapping(raw).rows[("primary", "npo")]
    assert row.funnel.protection_feasible_all_arms == 0
    assert row.funnel.protection_common == 0


def test_unplanned_keys_and_changed_frozen_selection_are_rejected():
    plan = _plan(units=[("r1", "1")])
    extra = _prediction("r2")
    with pytest.raises(EvidenceValidationError, match="unplanned unit"):
        aggregate_raw_evidence(plan, extra, [])

    changed = _prediction("r1")
    changed[0]["prediction_selection"] = _selection(0.9)
    with pytest.raises(EvidenceValidationError, match="frozen unit plan"):
        aggregate_raw_evidence(plan, changed, [])


def test_unequal_seed_counts_are_averaged_within_request_then_equally():
    plan = _plan(units=[("r1", "1"), ("r1", "2"), ("r2", "1")])
    first = _prediction("r1", "1")
    second = _prediction("r1", "2")
    third = _prediction("r2", "1")
    # Reverse joint only for r2. Request-equal rho is (1 + -1)/2 = 0;
    # pooling three seeds would incorrectly give 1/3.
    for row in third:
        row["joint"] = -row["joint"]
    raw = aggregate_raw_evidence(plan, first + second + third, [])
    row = EvidenceLedger.from_mapping(raw).rows[("primary", "npo")]
    assert row.prediction.joint_rho == pytest.approx(0.0)


def test_cli_writes_incomplete_rows_when_raw_shards_are_absent(tmp_path):
    plan_path = tmp_path / "plan.json"
    output = tmp_path / "ledger.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bootstrap": {"replicates": 21, "seed": 1},
                "units": [
                    {
                        "setting": "primary",
                        "parent": "npo",
                        "request": "r1",
                        "seed": "1",
                        "prediction_selection": _selection(),
                        "protection_selection": _selection(),
                        "repeated_random_draws": ["d0"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/paper/aggregate_raw.py"),
            "--plan",
            str(plan_path),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    row = EvidenceLedger.read(output).rows[("primary", "npo")]
    assert not row.attempted
    assert not row.completed
    assert row.funnel.trajectories_planned == 1


def test_tail_artifact_is_derived_from_damage_and_fails_closed(tmp_path):
    reference_metrics = {
        "rho_all": {"type": "number", "aggregate": "mean"},
        "rho_output": {"type": "number", "aggregate": "mean"},
        "rho_representation": {"type": "number", "aggregate": "mean"},
        "top_q_recall": {"type": "number", "aggregate": "mean"},
        "common_support": {"type": "integer", "aggregate": "sum"},
    }
    construction_metrics = {
        "joint_rho": {"type": "number", "aggregate": "mean"},
        "min_gain": {"type": "number", "aggregate": "mean"},
        "min_lower_bound": {"type": "number", "aggregate": "min"},
        "top_q_recall": {"type": "number", "aggregate": "mean"},
        "common_support": {"type": "integer", "aggregate": "sum"},
        "eligible": {"type": "boolean", "aggregate": "all_true"},
        "claim_pass": {"type": "boolean", "aggregate": "all_true"},
    }
    contract = {
        "tail_structure": {
            "kind": "tail_from_prediction",
            "rows": [{"setting": "primary", "parent": "npo"}],
            "q": 0.5,
            "cvar_q": 0.75,
            "permutations": 39,
            "supplementary_blocks": {
                "reference_roster": {
                    "key_fields": ["block", "row"],
                    "group_by": ["block", "row"],
                    "planned": [{"block": "reference_roster", "row": "exact"}],
                    "metrics": reference_metrics,
                },
                "construction_checks": {
                    "key_fields": ["block", "row"],
                    "group_by": ["block", "row"],
                    "planned": [{"block": "construction_checks", "row": "layer"}],
                    "metrics": construction_metrics,
                },
            },
        }
    }
    plan = replace(
        _plan(), artifact_contracts=contract, source_sha256="b" * 64
    )
    supplementary = {
        "tail_structure": [
            {
                "block": "reference_roster",
                "row": "exact",
                "rho_all": 0.5,
                "rho_output": 0.5,
                "rho_representation": 0.5,
                "top_q_recall": 0.5,
                "common_support": 2,
            },
            {
                "block": "construction_checks",
                "row": "layer",
                "joint_rho": 0.5,
                "min_gain": 0.1,
                "min_lower_bound": 0.01,
                "top_q_recall": 0.5,
                "common_support": 2,
                "eligible": True,
                "claim_pass": True,
            },
        ]
    }
    statuses, paths = build_raw_artifacts(
        plan,
        _prediction("r1") + _prediction("r2"),
        supplementary,
        output_dir=tmp_path,
    )
    assert statuses["tail_structure"]["completed"] is True
    artifact = json.loads(paths["tail_structure"].read_text(encoding="utf-8"))
    row = artifact["rows"][0]
    assert row["support"] == {"n": 2, "N": 2}
    assert row["mass_ratio"]["estimate"] > 1.0
    assert row["group_lift"]["estimate"] >= 0.0
    assert 0.0 < row["permutation_p"] <= 1.0
    ledger = aggregate_raw_evidence(
        plan,
        _prediction("r1") + _prediction("r2"),
        [],
        artifacts=statuses,
    )
    validate_artifact_files(EvidenceLedger.from_mapping(ledger), repository_root=ROOT)

    incomplete, _ = build_raw_artifacts(
        plan,
        _prediction("r1"),
        supplementary,
        output_dir=tmp_path / "incomplete",
    )
    assert incomplete["tail_structure"]["completed"] is False
    assert "headline_tex" not in incomplete["tail_structure"]


def _metric_specs(artifact_id: str) -> dict:
    if artifact_id == "lse_fidelity_cost":
        return {
            "rho_exact": {"type": "number", "aggregate": "mean"},
            "overlap_k": {"type": "number", "aggregate": "mean"},
            "split_half_rho": {"type": "number", "aggregate": "range"},
            "perturbation_survival": {"type": "number", "aggregate": "min"},
            "time_seconds": {"type": "number", "aggregate": "median"},
            "peak_memory_bytes": {"type": "integer", "aggregate": "max"},
            "integrity_valid": {"type": "boolean", "aggregate": "count_true"},
            "candidate_backward": {"type": "boolean", "aggregate": "all_true"},
        }
    if artifact_id == "protection_budget_sweep":
        return {
            "worst_effect": {"type": "number", "aggregate": "max"},
            "max_upper_bound": {"type": "number", "aggregate": "max"},
            "bottleneck": {"type": "text", "aggregate": "first"},
            "eligible": {"type": "boolean", "aggregate": "count_true"},
            "claim_pass": {"type": "boolean", "aggregate": "count_true"},
            "min_forget_margin": {"type": "number", "aggregate": "min"},
            "min_utility_margin": {"type": "number", "aggregate": "min"},
            "accepted_updates": {"type": "integer", "aggregate": "sum"},
            "common_support": {"type": "integer", "aggregate": "sum"},
            "random_draws_complete": {"type": "boolean", "aggregate": "all_true"},
        }
    return {
        "rho_g": {"type": "number", "aggregate": "mean"},
        "rho_h": {"type": "number", "aggregate": "mean"},
        "rho_joint": {"type": "number", "aggregate": "mean"},
        "top_q_lift": {"type": "number", "aggregate": "mean"},
        "displacement_matched": {"type": "boolean", "aggregate": "all_true"},
        "common_support": {"type": "integer", "aggregate": "sum"},
    }


def _measurement_record(artifact_id: str, request: str) -> dict:
    result = {"cell": "row0", "request": request, "seed": "1"}
    for field, spec in _metric_specs(artifact_id).items():
        if spec["type"] == "number":
            result[field] = 0.5
        elif spec["type"] == "integer":
            result[field] = 2
        elif spec["type"] == "boolean":
            result[field] = True
        else:
            result[field] = "joint/no_repair/mean"
    return result


def test_all_table_artifacts_have_exact_raw_schema_and_cell_sources(tmp_path):
    contracts = {}
    records = {}
    for artifact_id in (
        "lse_fidelity_cost",
        "protection_budget_sweep",
        "specificity_negative_controls",
    ):
        contracts[artifact_id] = {
            "kind": "measurements",
            "key_fields": ["cell", "request", "seed"],
            "group_by": ["cell"],
            "planned": [
                {"cell": "row0", "request": "r1", "seed": "1"},
                {"cell": "row0", "request": "r2", "seed": "1"},
            ],
            "metrics": _metric_specs(artifact_id),
        }
        records[artifact_id] = [
            _measurement_record(artifact_id, "r1"),
            _measurement_record(artifact_id, "r2"),
        ]
    plan = replace(_plan(), artifact_contracts=contracts)
    statuses, paths = build_raw_artifacts(
        plan,
        _prediction("r1") + _prediction("r2"),
        records,
        output_dir=tmp_path,
    )
    assert all(status["completed"] for status in statuses.values())
    for artifact_id, path in paths.items():
        artifact = json.loads(path.read_text(encoding="utf-8"))
        assert artifact["complete"]
        assert artifact["planned_units"] == 2
        assert artifact["observed_units"] == 2
        assert len(artifact["cells"][0]["source_keys"]) == 2

    bad = dict(records)
    bad["lse_fidelity_cost"] = [
        {**records["lse_fidelity_cost"][0], "request": "not-planned"}
    ]
    with pytest.raises(EvidenceValidationError, match="unplanned key"):
        build_raw_artifacts(
            plan,
            _prediction("r1") + _prediction("r2"),
            bad,
            output_dir=tmp_path / "bad",
        )


def test_measurement_contract_cannot_omit_a_physical_table_field(tmp_path):
    contract = {
        "lse_fidelity_cost": {
            "kind": "measurements",
            "key_fields": ["cell"],
            "group_by": ["cell"],
            "planned": [{"cell": "row0"}],
            "metrics": {
                key: value
                for key, value in _metric_specs("lse_fidelity_cost").items()
                if key != "peak_memory_bytes"
            },
        }
    }
    plan = replace(_plan(), artifact_contracts=contract)
    with pytest.raises(EvidenceValidationError, match="lacks table fields"):
        build_raw_artifacts(
            plan,
            _prediction("r1") + _prediction("r2"),
            {},
            output_dir=tmp_path,
        )
