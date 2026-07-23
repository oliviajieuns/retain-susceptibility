"""CPU contract tests for the alpha-runner -> paper raw bridge."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rsus.evidence.raw import raw_plan_from_mapping
from rsus.evidence.schemas import EvidenceValidationError


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_alpha_protection",
    ROOT / "experiments/paper/export_alpha_protection.py",
)
exporter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(exporter)


def _selection(alpha: float = 0.5) -> dict:
    return {"valid": True, "fallback": False, "alpha": alpha}


def _plan(request: str = "tofu-a188"):
    return raw_plan_from_mapping(
        {
            "schema_version": 1,
            "bootstrap": {"replicates": 5, "seed": 1},
            "units": [
                {
                    "setting": "tofu_qwen25_7b",
                    "parent": "npo",
                    "request": request,
                    "seed": 2025,
                    "prediction_selection": _selection(0.4),
                    "protection_selection": _selection(0.5),
                    "repeated_random_draws": ["rand-000", "rand-001"],
                }
            ],
        }
    )


def _row(selector: str, *, alpha=None, deployed=False, draw_id=None, offset=0.0):
    slacks = {
        "direct_forgetting": 0.05,
        "paraphrase_forgetting": 0.04,
        "extraction_generation": 0.03,
        "utility": 0.02,
    }
    row = {
        "campaign_phase": "audit",
        "model_id": "qwen25_7b",
        "request": "tofu-a188",
        "seed": 2025,
        "parent": "npo",
        "selector": selector,
        "selector_type": "mixture" if alpha is not None else "none",
        "alpha": alpha,
        "deployed": deployed,
        "executed": True,
        "reached": True,
        "feasible": True,
        "diagnostic_only": False,
        "step": 20,
        "selected_checkpoint_step": 20,
        "candidate_damage": {"c0": 0.1 + offset, "c1": 0.2 + offset},
        "candidate_groups": {"c0": "g0", "c1": "g1"},
        "parent_checkpoint": {
            "first_direct_reaching": True,
            "step": 10,
            "block_sha256": "a" * 64,
            "shared_by_all_repair_arms": True,
        },
        "final_checkpoint_decision": {
            "feasible": True,
            "selected_index": 0,
            "selected_step": 20,
            "trace": [{"index": 0, "step": 20, "slacks": slacks}],
        },
    }
    if draw_id is not None:
        row.update(
            {
                "selector_type": "repeated_random_draw",
                "random_draw_id": draw_id,
            }
        )
    return row


def _payload(request: str = "tofu-a188", deployed_alpha: float = 0.5):
    draws = [
        _row("random", draw_id="rand-000", offset=0.3),
        _row("random", draw_id="rand-001", offset=0.4),
    ]
    random = _row("random", offset=0.35)
    random.update(
        {
            "selector_type": "repeated_random",
            "random_draws": draws,
            "random_draw_ids": ["rand-000", "rand-001"],
        }
    )
    rows = [
        _row("none", offset=0.5),
        _row("s_alpha_0p0", alpha=0.0, offset=0.4),
        _row("s_alpha_0p5", alpha=deployed_alpha, deployed=True),
        _row("s_alpha_1p0", alpha=1.0, offset=0.3),
        random,
        _row("exact_grad_norm", offset=-0.1),
    ]
    for row in rows:
        row["request"] = request
    for draw in draws:
        draw["request"] = request
    return {
        "manifest": {
            "campaign_phase": "audit",
            "model": "/models/Qwen2.5-7B-Instruct",
            "model_id": "qwen25_7b",
            "request": request,
            "seed": 2025,
        },
        "results": rows,
        "_source_path": "synthetic/results.json",
    }


def test_exporter_emits_exact_five_arm_candidate_schema():
    records = exporter.export_records(
        _plan(),
        [_payload()],
        setting="tofu_qwen25_7b",
        expected_model_source="/models/Qwen2.5-7B-Instruct",
    )
    # 2 candidates x (joint, no repair, S0, S1, 2 random draws)
    assert len(records) == 12
    assert {record["arm"] for record in records} == {
        "joint", "no_repair", "s0", "s1", "repeated_random"
    }
    random = [record for record in records if record["arm"] == "repeated_random"]
    assert {record["draw_id"] for record in random} == {"rand-000", "rand-001"}
    assert all(record["draw_complete"] for record in random)
    assert all(record["parent_checkpoint_id"] == "a" * 64 for record in records)
    assert all(record["extraction_generation_margin"] == pytest.approx(0.03) for record in records)


def test_exporter_rejects_old_runner_roster_instead_of_relabeling():
    with pytest.raises(EvidenceValidationError, match="refusing relabel"):
        exporter.export_records(
            _plan("tofu-a188"),
            [_payload("tofu-a181")],
            setting="tofu_qwen25_7b",
            expected_model_source="/models/Qwen2.5-7B-Instruct",
        )


def test_exporter_rejects_changed_selection_support_and_checkpoint():
    with pytest.raises(EvidenceValidationError, match="deployed alpha"):
        exporter.export_records(
            _plan(),
            [_payload(deployed_alpha=0.75)],
            setting="tofu_qwen25_7b",
            expected_model_source="/models/Qwen2.5-7B-Instruct",
        )

    support = _payload()
    support["results"][3]["candidate_damage"].pop("c1")
    support["results"][3]["candidate_groups"].pop("c1")
    with pytest.raises(EvidenceValidationError, match="exact candidate/group support"):
        exporter.export_records(
            _plan(),
            [support],
            setting="tofu_qwen25_7b",
            expected_model_source="/models/Qwen2.5-7B-Instruct",
        )

    checkpoint = _payload()
    checkpoint["results"][3]["parent_checkpoint"]["block_sha256"] = "b" * 64
    with pytest.raises(EvidenceValidationError, match="mix parent checkpoints"):
        exporter.export_records(
            _plan(),
            [checkpoint],
            setting="tofu_qwen25_7b",
            expected_model_source="/models/Qwen2.5-7B-Instruct",
        )
