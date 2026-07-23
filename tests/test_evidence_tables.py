"""CPU-only tests for the generated tab:core-evidence / tab:robustness bodies."""
from __future__ import annotations

from pathlib import Path

import pytest

from rsus.evidence.decisions import evaluate_evidence
from rsus.evidence.registry import load_contract
from rsus.evidence.schemas import EvidenceLedger
from rsus.evidence.tables import (
    render_core_evidence_table,
    render_robustness_table,
    write_tex_tables,
)

from test_paper_evidence import _config, _ledger, _row  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def _report(contract, ledger):
    return evaluate_evidence(contract, ledger)


def test_empty_ledger_renders_placeholders_only(tmp_path):
    contract = _config(tmp_path)
    ledger = EvidenceLedger.empty()
    report = _report(contract, ledger)
    core = render_core_evidence_table(contract, ledger, report)
    robustness = render_robustness_table(contract, ledger, report)
    assert r"\tblph" in core
    assert r"\label{tab:core-evidence}" in core
    assert r"\label{tab:robustness}" in robustness
    # No numeric prediction cell may appear without ledger evidence.
    assert "0.5" not in core.replace("0.05", "").replace("\\tabcolsep", "")
    assert "not attempted" in robustness


def test_passing_row_renders_bounds_and_yes_flags(tmp_path):
    contract = _config(tmp_path)
    ledger = _ledger([_row("primary")])
    report = _report(contract, ledger)
    core = render_core_evidence_table(contract, ledger, report)
    assert "0.500 [0.050]" in core  # joint rho with its one-sided LB
    assert "0.200 [0.050]" in core  # min endpoint gain
    assert "y/y" in core
    # Fidelity cells stay placeholders without a fidelity summary input.
    assert "$f_\\rho/f_K$" in core
    robustness = render_robustness_table(contract, ledger, report)
    assert "1/1" in robustness


def test_fidelity_summary_fills_rq2_cells(tmp_path):
    contract = _config(tmp_path)
    ledger = _ledger([_row("primary")])
    report = _report(contract, ledger)
    fidelity = {
        "primary": {
            "f_rho": 0.97,
            "f_k": 0.86,
            "f_rho_lb": 0.93,
            "f_k_lb": 0.79,
        }
    }
    core = render_core_evidence_table(contract, ledger, report, fidelity=fidelity)
    assert "0.97/0.86" in core
    assert "[+0.13/+0.09]" in core
    # RQ2 E/P becomes y/y: floors cleared, g_H and g_ctl bounds positive.
    assert core.count("y/y") >= 2


def test_failing_native_bound_blocks_rq3_pass(tmp_path):
    contract = _config(tmp_path)
    raw = _row("primary")
    raw["protection"]["native"]["s1"] = {
        "estimate": -0.02,
        "lower_bound": -0.05,
        "p_one_sided": 0.9,
    }
    ledger = _ledger([raw])
    report = _report(contract, ledger)
    decision = report["rows"][0]["protection"]
    assert decision["eligible"]
    assert not decision["claim_pass"]
    core = render_core_evidence_table(contract, ledger, report)
    assert "y/n" in core


def test_tail_coverage_below_080_blocks_rq1(tmp_path):
    contract = _config(tmp_path)
    raw = _row("primary")
    raw["prediction"]["tail_eligible_n"] = 1
    raw["prediction"]["tail_total_n"] = 2
    ledger = _ledger([raw])
    report = _report(contract, ledger)
    decision = report["rows"][0]["prediction"]
    assert decision["eligible"]
    assert not decision["claim_pass"]


def test_write_tex_tables_stays_inside_paper_root(tmp_path):
    contract = _config(tmp_path)
    ledger = _ledger([_row("primary")])
    report = _report(contract, ledger)
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "main.tex").write_text("x", encoding="utf-8")
    paths = write_tex_tables(contract, ledger, report, paper)
    assert len(paths) == 2
    for path in paths:
        assert path.is_file()
        assert str(path).startswith(str(paper.resolve()))


def test_repository_contract_renders_all_nine_settings():
    contract = load_contract(ROOT / "configs/paper/evidence.yaml")
    ledger = EvidenceLedger.empty()
    report = evaluate_evidence(contract, ledger)
    robustness = render_robustness_table(contract, ledger, report)
    for label in (
        "held-out TOFU requests",
        "WMDP-bio/MMLU",
        "MUSE-News",
        "RWKU",
        "MUSE-Books (stress)",
        "PISTOL (stress)",
        "Qwen2.5-1.5B (boundary)",
        "Qwen2.5-14B",
        "Llama-3.1-8B",
    ):
        assert label in robustness, label
    core = render_core_evidence_table(contract, ledger, report)
    for parent_label in ("GradDiff", "NPO", "SimNPO", "GRU", "RMU", "RepNoise", "CB"):
        assert parent_label in core
    assert "Output-readout parents" in core
    assert "Representation-readout parents" in core


def test_rq1_pass_requires_positive_joint_bound(tmp_path):
    contract = _config(tmp_path)
    raw = _row("primary")
    raw["prediction"]["joint"] = {
        "estimate": 0.5,
        "lower_bound": -0.01,
        "p_one_sided": 0.2,
    }
    ledger = _ledger([raw])
    report = _report(contract, ledger)
    decision = report["rows"][0]["prediction"]
    assert decision["eligible"]
    assert not decision["claim_pass"]
