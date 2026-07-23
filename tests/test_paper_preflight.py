"""CPU-only checks for paper campaign readiness before GPU allocation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/paper/preflight.py"


def _ready_contract(tmp_path: Path) -> tuple[Path, Path, Path]:
    evidence = yaml.safe_load(
        (ROOT / "configs/paper/evidence.yaml").read_text(encoding="utf-8")
    )
    evidence_path = tmp_path / "evidence.yaml"
    evidence_path.write_text(
        yaml.safe_dump(evidence, sort_keys=False), encoding="utf-8"
    )

    campaign = yaml.safe_load(
        (ROOT / "configs/paper/campaign.yaml").read_text(encoding="utf-8")
    )
    tofu = campaign["datasets"]["TOFU"]
    campaign["datasets"] = {
        setting["dataset"]: copy.deepcopy(tofu) for setting in evidence["settings"]
    }
    for model_name, model in campaign["models"].items():
        model_path = tmp_path / "models" / model_name
        model_path.mkdir(parents=True)
        model["source"] = str(model_path)
        model["source_kind"] = "local_path"
        model["provisioned"] = True
    executor = tmp_path / "paper_stage_executor.py"
    executor.write_text(
        "PAPER_STAGE_CONTRACT = "
        + repr(
            {
                "schema_version": 1,
                "stages": list(campaign["stages"]),
                "consumes_campaign_config": True,
                "uses_adapter_registry": True,
                "consumes_exact_roster": True,
                "emits_selection_inputs": True,
                "emits_candidate_level_prediction_raw": True,
                "emits_candidate_level_protection_raw": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for stage in campaign["stages"].values():
        stage["executor"] = str(executor)
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "freeze_id": "test-freeze",
        "source_campaign": campaign["campaign_id"],
        "frozen_before_target": True,
        "selections": {
            setting["id"]: {
                parent: {
                    "prediction": {"valid": True, "fallback": False, "alpha": 0.5},
                    "protection": {"valid": True, "fallback": False, "alpha": 0.5},
                }
                for parent in setting["parents"]
            }
            for setting in evidence["settings"]
        },
    }
    freeze_path = tmp_path / "selection_freeze.yaml"
    freeze_path.write_text(yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8")
    campaign["execution"]["selection_freeze"] = str(freeze_path)
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
    )
    return evidence_path, campaign_path, tmp_path / "report.json"


def _run(evidence: Path, campaign: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--evidence-config",
            str(evidence),
            "--campaign-config",
            str(campaign),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_ready_setting_reports_exact_rosters_dtype_and_seven_parents(tmp_path):
    evidence, campaign, output = _ready_contract(tmp_path)
    result = _run(evidence, campaign, output)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ready"]
    assert report["datasets"]["TOFU"]["pairwise_disjoint"]
    assert set(report["datasets"]["TOFU"]["rosters"]) == {
        "D_cal",
        "D_pred",
        "D_prot",
        "target",
    }
    assert all(
        roster["exact"] and len(roster["sha256"]) == 64
        for roster in report["datasets"]["TOFU"]["rosters"].values()
    )
    assert report["models"]["Qwen2.5-7B"]["dtype"] == "float32"
    assert report["models"]["Qwen2.5-7B"]["parents_available"] == 7
    assert set(report["required_parents"]) <= set(
        report["implemented_parent_objectives"]
    )
    assert report["summary"]["settings_ready"] == 8
    assert report["summary"]["stages_ready"] == 32
    assert report["summary"]["unready_executors"] == []
    assert report["summary"]["selection_freeze_ready"]


def test_roster_overlap_fails_every_stage(tmp_path):
    evidence, campaign_path, output = _ready_contract(tmp_path)
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign["datasets"]["TOFU"]["rosters"]["D_prot"][0] = "tofu-a180"
    campaign_path.write_text(
        yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
    )
    result = _run(evidence, campaign_path, output)
    assert result.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["datasets"]["TOFU"]["pairwise_disjoint"]
    assert report["datasets"]["TOFU"]["overlaps"] == {
        "D_pred__D_prot": ["tofu-a180"]
    }
    assert not any(
        stage["ready"] for stage in report["settings"][0]["stages"].values()
    )


def test_tbd_missing_adapter_and_unprovisioned_model_are_explicit(tmp_path):
    output = tmp_path / "repository-report.json"
    result = _run(
        ROOT / "configs/paper/evidence.yaml",
        ROOT / "configs/paper/campaign.yaml",
        output,
    )
    assert result.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "WMDP-bio/MMLU" in report["summary"]["missing_adapter_datasets"]
    assert "MUSE-News" in report["summary"]["missing_adapter_datasets"]
    assert "MUSE-Books" in report["summary"]["missing_adapter_datasets"]
    assert "PISTOL" in report["summary"]["missing_adapter_datasets"]
    # RWKU graduated from this list on 2026-07-23: the adapter is registered
    # and its rosters are concrete, pairwise-disjoint request ids.
    assert "RWKU" not in report["summary"]["missing_adapter_datasets"]
    assert report["datasets"]["RWKU"]["pairwise_disjoint"]
    for roster in report["datasets"]["RWKU"]["rosters"].values():
        assert "reasons" not in roster or not any(
            "unresolved" in reason for reason in roster["reasons"]
        )
    assert report["datasets"]["WMDP-bio/MMLU"]["rosters"]["D_cal"][
        "reasons"
    ] == ["contains unresolved ids: TBD_WMDP_D_CAL_REQUEST_IDS"]
    assert "Llama-3.1-8B" in report["summary"]["unprovisioned_models"]


def test_missing_one_parent_and_invalid_dtype_fail_closed(tmp_path):
    evidence, campaign_path, output = _ready_contract(tmp_path)
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    model = campaign["models"]["Qwen2.5-7B"]
    model["dtype"] = "auto"
    model["parents"]["rmu"] = False
    campaign_path.write_text(
        yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
    )
    result = _run(evidence, campaign_path, output)
    assert result.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    model_report = report["models"]["Qwen2.5-7B"]
    assert model_report["parents_available"] == 6
    assert any("dtype must be explicit" in reason for reason in model_report["reasons"])
    assert any("rmu" in reason for reason in model_report["reasons"])


def test_valid_bfloat16_still_violates_frozen_fp32_contract(tmp_path):
    evidence, campaign_path, output = _ready_contract(tmp_path)
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign["models"]["Qwen2.5-7B"]["dtype"] = "bfloat16"
    campaign_path.write_text(
        yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
    )
    result = _run(evidence, campaign_path, output)
    assert result.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert any(
        "violates confirmatory precision 'float32'" in reason
        for reason in report["models"]["Qwen2.5-7B"]["reasons"]
    )


@pytest.mark.parametrize(
    "entrypoint",
    [
        "experiments/channel_matrix/run_campaign.py",
        "experiments/channel_matrix/aggregate.py",
    ],
)
def test_legacy_runner_or_summary_aggregator_is_not_a_candidate_raw_exporter(
    tmp_path, entrypoint
):
    evidence, campaign_path, output = _ready_contract(tmp_path)
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign["stages"]["prediction"]["executor"] = str(
        ROOT / entrypoint
    )
    campaign_path.write_text(
        yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
    )
    result = _run(evidence, campaign_path, output)
    assert result.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    executor = report["executors"]["prediction"]
    assert not executor["ready"]
    assert any("lacks literal PAPER_STAGE_CONTRACT" in reason for reason in executor["reasons"])
    assert not any(
        setting["stages"]["prediction"]["ready"]
        for setting in report["settings"]
    )


def test_adapter_rejects_out_of_domain_roster_id(tmp_path):
    evidence, campaign_path, output = _ready_contract(tmp_path)
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    campaign["datasets"]["TOFU"]["rosters"]["target"][-1] = "tofu-a999"
    campaign_path.write_text(
        yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8"
    )
    result = _run(evidence, campaign_path, output)
    assert result.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["datasets"]["TOFU"]["rosters"]["target"]["reasons"] == [
        "ids rejected by adapter contract: tofu-a999"
    ]


def test_shortened_evidence_roster_is_a_contract_error(tmp_path):
    evidence, campaign_path, output = _ready_contract(tmp_path)
    raw = yaml.safe_load(evidence.read_text(encoding="utf-8"))
    raw["settings"].pop()
    evidence.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    result = _run(evidence, campaign_path, output)
    assert result.returncode == 1
    assert "must exactly match paper_contract.setting_ids" in result.stderr
    assert not output.exists()
