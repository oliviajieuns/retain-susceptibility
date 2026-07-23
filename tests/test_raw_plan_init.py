"""Adversarial checks for freezing the raw paper denominator."""
from __future__ import annotations

import copy
import json
import argparse
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from experiments.cost.bench import _cost_row, _protocol_sha256
from experiments.paper.init_raw_plan import build_plan, selection_template
from rsus.costs import CostRecord
from rsus.evidence.schemas import EvidenceValidationError


ROOT = Path(__file__).resolve().parents[1]


def _contracts():
    evidence = yaml.safe_load(
        (ROOT / "configs/paper/evidence.yaml").read_text(encoding="utf-8")
    )
    campaign = yaml.safe_load(
        (ROOT / "configs/paper/campaign.yaml").read_text(encoding="utf-8")
    )
    freeze = selection_template(evidence, campaign["campaign_id"])
    freeze.update(
        {
            "status": "frozen",
            "freeze_id": "dev-selection-2027-01",
            "frozen_before_target": True,
        }
    )
    primary = "tofu_qwen25_1p5b"
    for parent in freeze["selections"][primary].values():
        parent["prediction"] = {"valid": True, "fallback": False, "alpha": 0.4}
        parent["protection"] = {"valid": True, "fallback": False, "alpha": 0.6}
    return evidence, campaign, freeze


def test_primary_plan_is_complete_and_fidelity_keys_match_cost_rows():
    evidence, campaign, freeze = _contracts()
    plan = build_plan(
        evidence,
        campaign,
        freeze,
        setting_ids={"tofu_qwen25_1p5b"},
    )
    # 10 untouched requests x 2 seeds x 7 parents.
    assert len(plan["units"]) == 140
    assert plan["selection_freeze_id"] == "dev-selection-2027-01"
    assert set(plan["artifact_contracts"]) == {
        "campaign_manifest",
        "tail_structure",
        "lse_fidelity_cost",
        "protection_budget_sweep",
        "specificity_negative_controls",
    }

    fidelity = plan["artifact_contracts"]["lse_fidelity_cost"]
    assert fidelity["protocol"] == campaign["execution"]["fidelity"]
    assert fidelity["key_fields"] == [
        "model",
        "model_source",
        "precision",
        "block",
        "protocol_sha256",
        "profiler",
        "R",
        "repeat",
    ]
    # Two provisioned models x (exact + three R values) x three repeats.
    assert len(fidelity["planned"]) == 24
    planned = next(
        row
        for row in fidelity["planned"]
        if row["model"] == "Qwen2.5-1.5B"
        and row["profiler"] == "loss_shake"
        and row["R"] == 16
        and row["repeat"] == 0
    )
    protocol = campaign["execution"]["fidelity"]
    runner_args = argparse.Namespace(
        **{
            **protocol,
            "dirs": ",".join(str(value) for value in protocol["directions"]),
        }
    )
    assert _protocol_sha256(runner_args, protocol["directions"]) == planned[
        "protocol_sha256"
    ]
    emitted = _cost_row(
        model_id="Qwen2.5-1.5B",
        model_source=planned["model_source"],
        precision="float32",
        block=planned["block"],
        protocol_sha256=planned["protocol_sha256"],
        profiler="loss_shake",
        directions=16,
        repeat=0,
        rho_exact=0.9,
        overlap_k=0.8,
        split_half_rho=0.75,
        perturbation_survival=0.95,
        record=CostRecord(wall_s=1.0, peak_mem_bytes=1024),
        valid=True,
        candidate_backward=False,
    )
    assert {field: emitted[field] for field in fidelity["key_fields"]} == planned


def test_pending_freeze_and_unprovisioned_setting_fail_closed():
    evidence, campaign, freeze = _contracts()
    pending = copy.deepcopy(freeze)
    pending["freeze_id"] = "PENDING-DEVELOPMENT-SELECTIONS"
    with pytest.raises(EvidenceValidationError, match="non-PENDING"):
        build_plan(
            evidence,
            campaign,
            pending,
            setting_ids={"tofu_qwen25_1p5b"},
        )

    # Filling a selection file cannot silently turn an unavailable model into
    # an executable denominator.
    llama = "tofu_llama31_8b"
    for parent in freeze["selections"][llama].values():
        parent["prediction"] = {"valid": True, "fallback": False, "alpha": 0.5}
        parent["protection"] = {"valid": True, "fallback": False, "alpha": 0.5}
    with pytest.raises(EvidenceValidationError, match="not provisioned"):
        build_plan(evidence, campaign, freeze, setting_ids={llama})


def test_execution_protocol_rejects_cost_runner_drift():
    evidence, campaign, freeze = _contracts()
    campaign["execution"]["fidelity"]["directions"] = [15, 32]
    with pytest.raises(EvidenceValidationError, match="positive even"):
        build_plan(
            evidence,
            campaign,
            freeze,
            setting_ids={"tofu_qwen25_1p5b"},
        )

    _, campaign, freeze = _contracts()
    campaign["execution"]["fidelity"]["block_last_n"] = 29
    with pytest.raises(EvidenceValidationError, match="no larger than the model depth"):
        build_plan(
            evidence,
            campaign,
            freeze,
            setting_ids={"tofu_qwen25_1p5b"},
        )


def test_fallback_selection_is_frozen_but_never_mislabelled_valid():
    evidence, campaign, freeze = _contracts()
    parent = freeze["selections"]["tofu_qwen25_1p5b"]["npo"]
    parent["prediction"] = {"valid": False, "fallback": True, "alpha": 0.5}
    plan = build_plan(
        evidence,
        campaign,
        freeze,
        setting_ids={"tofu_qwen25_1p5b"},
    )
    npo = [unit for unit in plan["units"] if unit["parent"] == "npo"]
    assert npo
    assert all(
        unit["prediction_selection"]
        == {"valid": False, "fallback": True, "alpha": 0.5}
        for unit in npo
    )


def test_cli_entrypoint_writes_a_parseable_partial_plan(tmp_path):
    _evidence, _campaign, freeze = _contracts()
    freeze_path = tmp_path / "selection_freeze.yaml"
    output = tmp_path / "raw_plan.json"
    freeze_path.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/paper/init_raw_plan.py"),
            "--selection-freeze",
            str(freeze_path),
            "--setting",
            "tofu_qwen25_1p5b",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert len(raw["units"]) == 140
    assert raw["selection_freeze_id"] == "dev-selection-2027-01"
