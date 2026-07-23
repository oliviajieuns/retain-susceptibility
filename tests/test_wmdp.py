"""CPU-only tests for the WMDP-bio/MMLU adapter against the cached schemas."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from rsus.data.base import Request
from rsus.data.registry import get_adapter
from rsus.data.wmdp import (
    CHUNK_SIZE,
    NATIVE_AUDIT_STRIDE,
    QUESTIONS_TOTAL,
    REQUESTS_TOTAL,
    wmdp_request,
)

ROOT = Path(__file__).resolve().parents[1]


class MockTokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 50) for c in text[:40]]}


SUBJECTS = ("anatomy", "college_biology", "virology")  # sorted order


def fake_tables() -> dict[str, list[dict]]:
    """Mimic the cached cais/wmdp (wmdp-bio) and cais/mmlu (all) schemas."""
    wmdp = [
        {
            "question": f"Hazardous bio question {i}?",
            "choices": [f"choice {i}-{letter}" for letter in "abcd"],
            "answer": i % 4,
        }
        for i in range(QUESTIONS_TOTAL)
    ]
    mmlu = [
        {
            "question": f"{subject} question {i}?",
            "choices": [f"{subject} option {i}-{letter}" for letter in "abcd"],
            "answer": (i + 1) % 4,
            "subject": subject,
        }
        for subject in SUBJECTS
        for i in range(10)
    ]
    return {"wmdp_bio": wmdp, "mmlu": mmlu}


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_cached_arrows_filters_split_and_picks_newest_fingerprint(tmp_path):
    from rsus.data.wmdp import find_cached_arrows

    base = tmp_path / "datasets" / "cais___mmlu" / "all"
    old = base / "0.0.1" / "aaa"
    new = base / "0.0.1" / "bbb"
    for fingerprint in (old, new):
        fingerprint.mkdir(parents=True)
        (fingerprint / "dataset_info.json").write_text("{}", encoding="utf-8")
        for shard in ("mmlu-test.arrow", "mmlu-validation.arrow",
                      "mmlu-auxiliary_train-00000-of-00002.arrow"):
            (fingerprint / shard).write_text("", encoding="utf-8")
    arrows = find_cached_arrows("cais___mmlu", "all", "test", hf_home=tmp_path)
    assert arrows == [new / "mmlu-test.arrow"]
    assert find_cached_arrows("cais___wmdp", "wmdp-bio", "test", hf_home=tmp_path) == []


def test_wmdp_request_structure_matches_contract():
    request = wmdp_request(
        MockTokenizer(), request_index=0, candidate_subjects=[0, 2], tables=fake_tables()
    )
    assert isinstance(request, Request)
    assert request.request_id == "wmdp-r000"
    # forget = one frozen 40-question slice, QA-rendered with the correct choice
    assert len(request.forget) == CHUNK_SIZE
    assert all(e.group == "slice-000" for e in request.forget)
    for e in request.forget:
        row_idx = int(e.example_id.rsplit("-q", 1)[1])
        assert e.text == (f"Question: Hazardous bio question {row_idx}?\n"
                          f"Answer: choice {row_idx}-{'abcd'[row_idx % 4]}")
    # universe = first 8 questions of each pooled subject, subject-grouped
    groups = {e.group for e in request.universe.examples}
    assert groups == {"mmlu-anatomy", "mmlu-virology"}
    assert len(request.universe) == 2 * 8
    # native audit = every 4th candidate of the frozen universe
    expected_native = {
        e.example_id for i, e in enumerate(request.universe.examples)
        if i % NATIVE_AUDIT_STRIDE == 0
    }
    assert request.native_audit_ids == expected_native and expected_native
    assert request.forget_sha and request.universe.sha
    # no forget probe leaks into the retained universe
    assert not {e.example_id for e in request.forget} & {
        e.example_id for e in request.universe.examples
    }


def test_wmdp_forget_slices_are_deterministic_and_disjoint():
    tables = fake_tables()
    first = wmdp_request(MockTokenizer(), 0, [0, 1], tables=tables)
    again = wmdp_request(MockTokenizer(), 0, [0, 1], tables=tables)
    assert [e.example_id for e in first.forget] == [e.example_id for e in again.forget]
    assert first.forget_sha == again.forget_sha
    assert first.universe.sha == again.universe.sha
    assert first.native_audit_ids == again.native_audit_ids

    other = wmdp_request(MockTokenizer(), 1, [0, 1], tables=tables)
    rows = lambda req: {e.example_id.rsplit("-q", 1)[1] for e in req.forget}  # noqa: E731
    assert not rows(first) & rows(other)
    # different slices share the frozen MMLU candidate universe
    assert first.universe.sha == other.universe.sha


def test_wmdp_universe_cap_is_deterministic_truncation():
    request = wmdp_request(
        MockTokenizer(), 0, [0, 1, 2], per_subject=10, max_candidates=12,
        tables=fake_tables(),
    )
    assert len(request.universe) == 12
    assert {e.group for e in request.universe.examples} == {"mmlu-anatomy", "mmlu-college_biology"}


def test_wmdp_request_rejects_bad_inputs():
    tables = fake_tables()
    with pytest.raises(ValueError, match="outside 0"):
        wmdp_request(MockTokenizer(), REQUESTS_TOTAL, [0], tables=tables)
    with pytest.raises(ValueError, match="at least one"):
        wmdp_request(MockTokenizer(), 0, [], tables=tables)
    with pytest.raises(ValueError, match="duplicates"):
        wmdp_request(MockTokenizer(), 0, [1, 1], tables=tables)
    with pytest.raises(ValueError, match="subject indices outside"):
        wmdp_request(MockTokenizer(), 0, [0, 99], tables=tables)
    truncated = {"wmdp_bio": tables["wmdp_bio"][:100], "mmlu": tables["mmlu"]}
    with pytest.raises(ValueError, match="expected 1273"):
        wmdp_request(MockTokenizer(), 0, [0], tables=truncated)


def test_wmdp_registry_adapter_and_roster_ids():
    adapter = get_adapter("WMDP-bio/MMLU")
    assert adapter.key == "wmdp_bio_mmlu"
    assert get_adapter("cais/wmdp") is adapter and get_adapter("wmdp-bio") is adapter
    for stage in ("calibration", "prediction", "protection", "target_evaluation"):
        assert adapter.capabilities.supports(stage)
    assert adapter.capabilities.native_audit
    validator = adapter.roster_id_validator
    assert validator("wmdp-r000") and validator(f"wmdp-r{REQUESTS_TOTAL - 1:03d}")
    assert not validator(f"wmdp-r{REQUESTS_TOTAL:03d}")
    assert not validator("rwku-t000") and not validator("wmdp-x1")

    request = adapter.build_request(
        tokenizer=MockTokenizer(), request_index=1, candidate_subjects=[0, 2],
        tables=fake_tables(),
    )
    assert request.request_id == "wmdp-r001"


def test_wmdp_campaign_config_builds_wmdp_gate_commands():
    campaign = _load_module("wmdp_campaign", "experiments/channel_matrix/run_campaign.py")
    cfg = yaml.safe_load(
        (ROOT / "configs/channel_matrix/wmdp_7b.yaml").read_text(encoding="utf-8")
    )
    campaign._validate_campaign(cfg)
    assert campaign._request_dirname(cfg, 0) == "wmdp-r000"
    assert campaign._request_dirname({"dataset": "rwku"}, 0) == "rwku-t000"
    assert campaign._request_dirname({"dataset": "tofu"}, 198) == "tofu-a198"

    models = [m for m in cfg["models"] if m.get("enabled", True)]
    commands = list(
        campaign.calibration_commands(cfg, models, ROOT / "runs/channel_matrix_wmdp7b")
    )
    n_settings = sum(len(s) for s in cfg["calibration"]["objective_grid"].values())
    assert len(commands) == len(cfg["calibration"]["authors"]) * n_settings
    for out, cmd in commands:
        assert cmd[cmd.index("--dataset") + 1] == "wmdp_bio_mmlu"
        assert "wmdp-r00" in str(out)
        assert "idkdpo" not in cmd  # excluded roster, parallel with RWKU

    fidelity = list(
        campaign.fidelity_commands(cfg, models, ROOT / "runs/channel_matrix_wmdp7b")
    )
    for _csv, _cert, cmd in fidelity:
        assert cmd[cmd.index("--dataset") + 1] == "wmdp_bio_mmlu"


def test_fd_fidelity_builds_pools_through_registry_adapters():
    fidelity = _load_module("wmdp_fd_fidelity", "experiments/diag/fd_fidelity.py")

    request = fidelity.build_fidelity_request(
        "wmdp_bio_mmlu", MockTokenizer(), 0, universe_authors=2,
        candidate_authors=[0, 2], tables=fake_tables(),
    )
    assert request.request_id == "wmdp-r000"
    assert len(request.universe) == 2 * 8

    from test_rwku import fake_tables as rwku_fake_tables

    rwku = fidelity.build_fidelity_request(
        "rwku", MockTokenizer(), 0, universe_authors=2,
        candidate_authors=[1, 2], tables=rwku_fake_tables(),
    )
    assert rwku.request_id == "rwku-t000"

    with pytest.raises(ValueError, match="requires an explicit frozen"):
        fidelity.build_fidelity_request(
            "wmdp_bio_mmlu", MockTokenizer(), 0, universe_authors=2,
            candidate_authors=None, tables=fake_tables(),
        )
