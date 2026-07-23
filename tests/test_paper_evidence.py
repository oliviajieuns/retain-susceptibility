"""CPU-only tests for the fail-closed paper evidence pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from rsus.evidence.decisions import evaluate_evidence
from rsus.evidence.registry import load_contract
from rsus.evidence.rendering import render_tex_macros, write_tex_macros
from rsus.evidence.schemas import (
    EvidenceLedger,
    EvidenceValidationError,
    validate_artifact_files,
)
from rsus.evidence.statistics import (
    finite_sample_one_sided_p,
    intersection_union_p,
    paired_differences,
    summarize_bootstrap_effect,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path):
    raw = {
        "schema_version": 1,
        "ledger": "ledger.json",
        "outputs": {
            "readiness_json": "readiness.json",
            "tex_macros": "sections/generated/results_macros.tex",
        },
        "decision": {
            "alpha": 0.05,
            "minimum_support_units": 2,
            "prediction_iut": {
                "contrasts": ["joint_minus_s0", "joint_minus_s1"],
                "favorable_sign": "positive",
            },
            "protection_iut": {
                "comparators": ["no_repair", "repeated_random", "s0", "s1"],
                "outcomes": ["mean", "cvar95"],
                "favorable_sign": "negative",
                "common_arms": ["joint", "no_repair", "repeated_random", "s0", "s1"],
                "exact_norm_role": "descriptive_same_estimand_reference_outside_iut",
            },
        },
        "settings": [
            {"id": "primary", "dataset": "D0", "model": "M0", "role": "primary", "parents": ["p"]},
            {"id": "model_a", "dataset": "D0", "model": "M1", "role": "model_scale", "parents": ["p"]},
            {"id": "model_b", "dataset": "D0", "model": "M2", "role": "model_family", "parents": ["p"]},
            {"id": "data_a", "dataset": "D1", "model": "M0", "role": "dataset_replication", "parents": ["p"]},
            {"id": "data_b", "dataset": "D2", "model": "M0", "role": "dataset_replication", "parents": ["p"]},
            {"id": "data_c", "dataset": "D3", "model": "M0", "role": "dataset_replication", "parents": ["p"]},
            {"id": "stress", "dataset": "DS", "model": "M0", "role": "stress", "parents": ["p"]},
        ],
        "multi_setting_rule": {
            "id": "test-rule",
            "primary_required": ["primary"],
            "groups": [
                {"id": "models", "settings": ["model_a", "model_b"], "minimum_pass": 1},
                {"id": "datasets", "settings": ["data_a", "data_b", "data_c"], "minimum_pass": 2},
            ],
            "parent_groups": [
                {"id": "all", "parents": ["p"], "minimum_joint_pass": 1},
            ],
            "stress_excluded": ["stress"],
            "require_both_claims": True,
        },
        "artifacts": [
            "campaign_manifest",
            "tail_structure",
            "lse_fidelity_cost",
        ],
        "tables": [
            {
                "id": "main_core_evidence",
                "label": "tab:core-evidence",
                "location": "main",
                "settings": ["primary"],
                "claims": ["prediction", "protection"],
                "artifacts": [],
                "producer": "builder",
            },
            {
                "id": "main_robustness",
                "label": "tab:robustness",
                "location": "main",
                "settings": [
                    "primary", "model_a", "model_b", "data_a", "data_b", "data_c", "stress"
                ],
                "claims": ["prediction", "protection"],
                "artifacts": [],
                "producer": "builder",
            },
            {
                "id": "appendix_scope_contract",
                "label": "tab:datasets",
                "location": "appendix",
                "settings": [],
                "claims": [],
                "artifacts": ["campaign_manifest"],
                "producer": "manifest",
            },
        ],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_contract(path)


def _effect(*, gain: bool) -> dict[str, float]:
    if gain:
        return {"estimate": 0.2, "lower_bound": 0.05, "p_one_sided": 0.01}
    return {"estimate": -0.2, "upper_bound": -0.05, "p_one_sided": 0.01}


def _row(setting: str, *, feasible: bool = True, complete: bool = True) -> dict:
    n = 2
    attempted_n = n if complete else 1
    completed_n = n if complete else 1
    reached_n = completed_n
    feasible_n = reached_n if feasible else max(0, reached_n - 1)
    return {
        "setting": setting,
        "parent": "p",
        "attempted": True,
        "completed": complete,
        "prediction_selection": {"valid": True, "fallback": False, "alpha": 0.5},
        "protection_selection": {"valid": True, "fallback": False, "alpha": 0.5},
        "funnel": {
            "profiles_planned": n,
            "profiles_valid": n,
            "trajectories_planned": n,
            "trajectories_attempted": attempted_n,
            "trajectories_completed": completed_n,
            "trajectories_reached": reached_n,
            "reached_with_valid_profile": reached_n,
            "prediction_common": reached_n,
            "protection_feasible_all_arms": feasible_n,
            "protection_common": feasible_n,
        },
        "prediction": {
            "paired": True,
            "joint_rho": 0.5,
            "top_q_recall": 0.7,
            "vs_s0": _effect(gain=True),
            "vs_s1": _effect(gain=True),
        },
        "protection": {
            "paired": True,
            "comparisons": {
                comparator: {
                    "mean": _effect(gain=False),
                    "cvar95": _effect(gain=False),
                }
                for comparator in ("no_repair", "repeated_random", "s0", "s1")
            },
            # Deliberately adverse: exact norm is descriptive and outside IUT.
            "exact_norm": {
                "mean": {"estimate": 1.0, "upper_bound": 2.0, "p_one_sided": 1.0},
                "cvar95": {"estimate": 1.0, "upper_bound": 2.0, "p_one_sided": 1.0},
            },
            "min_forget_margin": 0.1,
            "min_utility_margin": 0.1,
        },
    }


def _ledger(rows: list[dict], artifacts: dict | None = None) -> EvidenceLedger:
    return EvidenceLedger.from_mapping(
        {"schema_version": 1, "rows": rows, "artifacts": artifacts or {}}
    )


def test_one_sided_signs_and_iut_are_explicit():
    positive = summarize_bootstrap_effect(
        0.3, [0.1, 0.2, 0.3, 0.4], beneficial="positive", alpha=0.25
    )
    negative = summarize_bootstrap_effect(
        -0.3, [-0.5, -0.4, -0.3, -0.2], beneficial="negative", alpha=0.25
    )
    assert positive["lower_bound"] > 0
    assert negative["upper_bound"] < 0
    assert finite_sample_one_sided_p([1.0, 1.0], beneficial="positive") == pytest.approx(1 / 3)
    assert intersection_union_p([0.01, 0.04, 0.02]) == 0.04
    assert paired_differences({"u1": 2.0, "u2": 5.0}, {"u1": 1.0, "u2": 3.0}) == {
        "u1": 1.0,
        "u2": 2.0,
    }
    with pytest.raises(ValueError, match="identical unit keys"):
        paired_differences({"u1": 2.0}, {"u2": 1.0})


def test_missing_rows_remain_in_denominator_and_cannot_pass(tmp_path):
    contract = _config(tmp_path)
    report = evaluate_evidence(contract, _ledger([_row("primary")]))
    assert report["denominators"] == {
        "planned_rows": 7,
        "attempted_rows": 1,
        "completed_rows": 1,
        "missing_rows": 6,
    }
    missing = next(row for row in report["rows"] if row["setting"] == "model_a")
    assert not missing["prediction"]["claim_pass"]
    assert not missing["protection"]["claim_pass"]
    assert not report["multi_setting"]["pass"]


def test_protection_requires_all_five_arms_but_exact_norm_is_outside_iut(tmp_path):
    contract = _config(tmp_path)
    passing = evaluate_evidence(contract, _ledger([_row("primary")]))
    decision = passing["rows"][0]["protection"]
    assert decision["eligible"]
    assert decision["statistical_pass"]
    assert decision["claim_pass"]

    infeasible = evaluate_evidence(
        contract, _ledger([_row("primary", feasible=False)])
    )
    decision = infeasible["rows"][0]["protection"]
    assert decision["statistical_pass"]
    assert not decision["eligible"]
    assert not decision["claim_pass"]
    assert "not all five claim arms are feasible" in decision["reasons"]


def test_incomplete_trajectory_cannot_pass_even_with_favorable_effects(tmp_path):
    contract = _config(tmp_path)
    report = evaluate_evidence(contract, _ledger([_row("primary", complete=False)]))
    row = report["rows"][0]
    assert row["prediction"]["statistical_pass"]
    assert not row["prediction"]["claim_pass"]
    assert row["protection"]["statistical_pass"]
    assert not row["protection"]["claim_pass"]


def test_unpaired_summaries_fail_closed(tmp_path):
    contract = _config(tmp_path)
    raw = _row("primary")
    raw["prediction"]["paired"] = False
    raw["protection"]["paired"] = False
    report = evaluate_evidence(contract, _ledger([raw]))
    row = report["rows"][0]
    assert not row["prediction"]["data_complete"]
    assert not row["prediction"]["claim_pass"]
    assert not row["protection"]["data_complete"]
    assert not row["protection"]["claim_pass"]


def test_explicit_multi_setting_threshold_and_stress_exclusion(tmp_path):
    contract = _config(tmp_path)
    # The rule passes with primary + one model transfer + two dataset
    # replications. Missing model_b/data_c/stress remain visible failures.
    rows = [_row(name) for name in ("primary", "model_a", "data_a", "data_b")]
    report = evaluate_evidence(contract, _ledger(rows))
    assert report["denominators"]["planned_rows"] == 7
    assert report["denominators"]["completed_rows"] == 4
    assert report["multi_setting"]["pass"]
    assert report["multi_setting"]["stress_excluded"] == ["stress"]


def test_setting_chain_requires_same_parent_to_pass_both_claims(tmp_path):
    contract = _config(tmp_path)
    raw = _row("primary")
    raw["protection"]["comparisons"]["s1"]["mean"] = {
        "estimate": 0.1,
        "upper_bound": 0.2,
        "p_one_sided": 0.9,
    }
    report = evaluate_evidence(contract, _ledger([raw]))
    assert report["settings"]["primary"]["prediction"]["passed"] == 1
    assert report["settings"]["primary"]["protection"]["passed"] == 0
    assert not report["settings"]["primary"]["chain"]["pass"]


def test_macro_renderer_uses_exact_names_and_keeps_incomplete_placeholders(tmp_path):
    contract = _config(tmp_path)
    ledger = EvidenceLedger.empty()
    report = evaluate_evidence(contract, ledger)
    rendered = render_tex_macros(contract, ledger, report)
    for name in (
        "TailHeadline",
        "PredictionHeadline",
        "FidelityHeadline",
        "ProtectionHeadline",
        "TransferHeadline",
    ):
        assert rf"\renewcommand{{\{name}}}" in rendered
    assert rendered.count(r"\resph{") == 5

    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text("test", encoding="utf-8")
    target = write_tex_macros(contract, ledger, report, paper)
    assert target == paper / "sections/generated/results_macros.tex"
    assert target.read_text(encoding="utf-8") == rendered


def test_completed_artifact_hash_is_verified(tmp_path):
    artifact = tmp_path / "tail.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "tail_structure",
                "kind": "tail_from_prediction",
                "complete": True,
                "source_plan_sha256": "a" * 64,
                "rows": [{"setting": "primary", "parent": "p"}],
                "supplementary": {
                    "reference_roster": {"complete": True},
                    "construction_checks": {"complete": True},
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    ledger = _ledger(
        [],
        {
            "tail_structure": {
                "completed": True,
                "path": str(artifact),
                "sha256": digest,
                "headline_tex": "Tail evidence is complete.",
            }
        },
    )
    validate_artifact_files(ledger, repository_root=ROOT)

    arbitrary = tmp_path / "arbitrary.json"
    arbitrary.write_text("{}\n", encoding="utf-8")
    arbitrary_ledger = _ledger(
        [],
        {
            "tail_structure": {
                "completed": True,
                "path": str(arbitrary),
                "sha256": hashlib.sha256(arbitrary.read_bytes()).hexdigest(),
            }
        },
    )
    with pytest.raises(EvidenceValidationError, match="schema/artifact_id"):
        validate_artifact_files(arbitrary_ledger, repository_root=ROOT)
    bad = json.loads(json.dumps({
        "schema_version": 1,
        "rows": [],
        "artifacts": {
            "tail_structure": {
                "completed": True,
                "path": str(artifact),
                "sha256": "0" * 64,
            }
        },
    }))
    with pytest.raises(EvidenceValidationError, match="sha256 mismatch"):
        validate_artifact_files(
            EvidenceLedger.from_mapping(bad), repository_root=ROOT
        )


def test_repository_registry_covers_two_main_and_five_appendix_tables():
    contract = load_contract(ROOT / "configs/paper/evidence.yaml")
    assert len(contract.planned_keys) == 8 * 7
    locations = [table.location for table in contract.tables.values()]
    assert locations.count("main") == 2
    assert locations.count("appendix") == 5
    assert contract.tables["main_core_evidence"].claims == (
        "prediction",
        "protection",
    )


def test_setting_breadth_rule_corrects_parent_selection_within_readout_group():
    contract = load_contract(ROOT / "configs/paper/evidence.yaml")
    report = evaluate_evidence(contract, EvidenceLedger.empty())
    groups = {
        group["id"]: group
        for group in report["settings"]["tofu_qwen25_1p5b"]["chain"][
            "parent_groups"
        ]
    }
    assert groups["output_readout"]["multiplicity"] == "bonferroni"
    assert groups["output_readout"]["per_parent_iut_alpha"] == pytest.approx(
        0.05 / 4
    )
    assert groups["representation_readout"][
        "per_parent_iut_alpha"
    ] == pytest.approx(0.05 / 3)


def test_registry_rejects_silent_iut_contract_drift(tmp_path):
    contract = _config(tmp_path)
    assert contract.alpha == 0.05
    path = tmp_path / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["decision"]["protection_iut"]["comparators"] = ["no_repair", "s0"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="comparators"):
        load_contract(path)


def test_cli_emits_fail_closed_readiness_and_macros(tmp_path):
    readiness = tmp_path / "readiness.json"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text("test", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/paper/build_evidence.py"),
            "--ledger",
            str(tmp_path / "missing-ledger.json"),
            "--readiness-out",
            str(readiness),
            "--paper-root",
            str(paper),
            "--require-ready",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    report = json.loads(readiness.read_text(encoding="utf-8"))
    assert report["denominators"]["planned_rows"] == 56
    assert report["denominators"]["completed_rows"] == 0
    macros = (paper / "sections/generated/results_macros.tex").read_text(
        encoding="utf-8"
    )
    assert macros.count(r"\resph{") == 5
