"""CPU-only tests for the agent-facing next-actions oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "cluster"))

import next_actions  # noqa: E402


def _root(tmp_path: Path, *, objective: str, alpha: str, provisioned: bool = True) -> Path:
    """Build a minimal campaign tree with controllable freeze states."""
    root = tmp_path / "repo"
    cfg_dir = root / "configs" / "channel_matrix"
    cfg_dir.mkdir(parents=True)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    if provisioned:
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

    def freeze(name: str, state: str) -> None:
        payload = {"status": state}
        if state == "frozen":
            payload["frozen_before_audit"] = True
        (cfg_dir / name).write_text(yaml.safe_dump(payload), encoding="utf-8")

    freeze("objective_freeze.yaml", objective)
    freeze("alpha_freeze.yaml", alpha)
    (cfg_dir / "camp.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"path": str(model_dir)},
                "audit": {"objective_freeze": "objective_freeze.yaml"},
                "alpha_protection": {"alpha_freeze": "alpha_freeze.yaml"},
            }
        ),
        encoding="utf-8",
    )
    return root


def _report(root: Path) -> dict:
    return next_actions.campaign_report(
        root, "camp", "configs/channel_matrix/camp.yaml", "runs/cluster_queue/q"
    )


def test_draft_objective_freeze_allows_only_prefreeze_lanes(tmp_path):
    report = _report(_root(tmp_path, objective="draft", alpha="draft"))
    assert report["allowed_now"] == ["fidelity", "calibration"]
    assert any("audit blocked" in line for line in report["blocked"])
    assert any("select-freeze" in line for line in report["human_next"])


def test_frozen_objective_with_draft_alpha_opens_audit_and_alpha_dev(tmp_path):
    report = _report(_root(tmp_path, objective="frozen", alpha="draft"))
    assert report["allowed_now"] == ["audit", "alpha-development"]
    assert any("alpha-audit blocked" in line for line in report["blocked"])


def test_both_frozen_opens_alpha_audit(tmp_path):
    report = _report(_root(tmp_path, objective="frozen", alpha="frozen"))
    assert report["allowed_now"] == ["audit", "alpha-audit"]
    assert report["blocked"] == []
    assert report["human_next"] == []


def test_unprovisioned_model_blocks_everything_but_reports_freezes(tmp_path):
    report = _report(
        _root(tmp_path, objective="frozen", alpha="frozen", provisioned=False)
    )
    assert report["allowed_now"] == []
    assert any("not provisioned" in line for line in report["blocked"])
    # Freeze visibility survives so an agent can plan the whole chain.
    assert report["objective_freeze"]["state"] == "frozen"


def test_missing_config_is_a_human_step(tmp_path):
    report = next_actions.campaign_report(
        tmp_path, "ghost", "configs/channel_matrix/ghost.yaml", "runs/q"
    )
    assert report["allowed_now"] == []
    assert any("HUMAN step" in line for line in report["blocked"])


def test_repository_report_runs_and_serializes(tmp_path):
    # The oracle must never mutate anything; it only reads and reports.
    report = next_actions.build_report(ROOT)
    text = json.dumps(report)
    assert "campaigns" in report and len(report["campaigns"]) >= 6
    assert "tofu_qwen25_7b" in text
