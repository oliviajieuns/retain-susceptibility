"""CPU-only tests for the RWKU adapter against the probed cache schema."""
from __future__ import annotations

import pytest
import torch

from rsus.data.base import Request
from rsus.data.registry import get_adapter
from rsus.data.rwku import (
    TARGETS_TOTAL,
    format_cloze,
    rwku_request,
)
from rsus.losses import IGNORE


class MockTokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 50) for c in text[:40]]}


def fake_tables() -> dict[str, list[dict]]:
    """Mimic the probed jinzhuoran/RWKU schema with three targets."""
    names = {0: "Stephen King", 1: "Ada Lovelace", 2: "Marie Curie"}
    targets = [
        {"target": names.get(i, f"Person {i}"), "intro": f"Please forget {names.get(i, i)}."}
        for i in range(TARGETS_TOTAL)
    ]
    forget1, forget2, neigh1, neigh2 = [], [], [], []
    for i in (0, 1, 2):
        subject = names[i]
        forget1.append({"query": f"{subject} is famous as a ___", "answer": "figure",
                        "subject": subject, "type": "cloze", "level": "1"})
        forget1.append({"query": f"{subject} was born in ___", "answer": "history",
                        "subject": subject, "type": "cloze", "level": "1"})
        forget2.append({"query": f"What is {subject} known for?", "answer": "Something notable.",
                        "subject": subject, "type": "simple question", "level": "2"})
        neigh1.append({"query": f"A work related to {subject} is ___", "answer": "famous",
                       "subject": subject, "neighbor": f"{subject} Work", "type": "cloze",
                       "level": "1"})
        neigh2.append({"query": f"Who collaborated with {subject}?", "answer": "A peer.",
                       "subject": subject, "neighbor": f"{subject} Peer",
                       "type": "simple question", "level": "2"})
    return {
        "forget_target": targets,
        "forget_level1": forget1,
        "forget_level2": forget2,
        "neighbor_level1": neigh1,
        "neighbor_level2": neigh2,
    }


def test_format_cloze_masks_prompt_and_appends_eos():
    ids, labels = format_cloze("Stephen King is an American ___", "author", MockTokenizer())
    n_prompt = int((labels == IGNORE).sum())
    assert n_prompt > 0
    assert labels[n_prompt:].tolist() == ids[n_prompt:].tolist()
    assert labels[-1].item() == MockTokenizer.eos_token_id


def test_rwku_request_structure_matches_contract():
    request = rwku_request(
        MockTokenizer(), target_index=0, candidate_targets=[1, 2], tables=fake_tables()
    )
    assert isinstance(request, Request)
    assert request.request_id == "rwku-t000"
    # forget = the target's own level-1/2 probes
    assert len(request.forget) == 3
    assert all(e.group == "target-000" for e in request.forget)
    # universe = own neighbors (native audit) + other targets' probes
    groups = {e.group for e in request.universe.examples}
    assert {"neighbor-Stephen King Work", "neighbor-Stephen King Peer",
            "target-001", "target-002"} == groups
    assert len(request.native_audit_ids) == 2
    assert request.native_audit_ids <= {e.example_id for e in request.universe.examples}
    assert request.forget_sha and request.universe.sha
    # no forget probe leaks into the retained universe
    assert not {e.example_id for e in request.forget} & {
        e.example_id for e in request.universe.examples
    }


def test_rwku_request_rejects_bad_pools():
    tables = fake_tables()
    with pytest.raises(ValueError, match="cannot appear in its own"):
        rwku_request(MockTokenizer(), 0, [0, 1], tables=tables)
    with pytest.raises(ValueError, match="at least one"):
        rwku_request(MockTokenizer(), 0, [], tables=tables)
    with pytest.raises(ValueError, match="duplicates"):
        rwku_request(MockTokenizer(), 0, [1, 1], tables=tables)
    with pytest.raises(ValueError, match="no probes"):
        rwku_request(MockTokenizer(), 0, [1, 150], tables=tables)


def test_rwku_registry_adapter_and_roster_ids():
    adapter = get_adapter("jinzhuoran/RWKU")
    assert adapter.key == "rwku"
    for stage in ("calibration", "prediction", "protection", "target_evaluation"):
        assert adapter.capabilities.supports(stage)
    assert adapter.capabilities.native_audit
    validator = adapter.roster_id_validator
    assert validator("rwku-t000") and validator("rwku-t199")
    assert not validator("rwku-t200") and not validator("tofu-a181") and not validator("rwku-x1")

    request = adapter.build_request(
        tokenizer=MockTokenizer(), target_index=1, candidate_targets=[0, 2],
        tables=fake_tables(),
    )
    assert request.request_id == "rwku-t001"


def test_paper_campaign_rwku_rosters_are_valid_and_disjoint():
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/paper/campaign.yaml").read_text(
            encoding="utf-8"
        )
    )
    rosters = cfg["datasets"]["RWKU"]["rosters"]
    validator = get_adapter("rwku").roster_id_validator
    seen: set[str] = set()
    for name, ids in rosters.items():
        assert ids, f"empty roster {name}"
        for rid in ids:
            assert validator(rid), f"invalid roster id {rid}"
        assert not seen & set(ids), f"roster {name} overlaps an earlier roster"
        seen |= set(ids)
