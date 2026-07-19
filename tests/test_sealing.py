"""Seal ledger discipline (DESIGN.md §6, §7 item 12)."""
import pytest

from rsus.sealing import SealedError, read_scores, seal_scores, unseal

SCORES = {"c00": 0.5, "c01": -0.1}


def test_seal_unseal_flow(tmp_path):
    seals, ledger = tmp_path / "seals", tmp_path / "ledger.jsonl"
    sha = seal_scores(seals, ledger, "r1", "fd", SCORES)
    assert len(sha) == 64

    # sealed: reading refuses
    with pytest.raises(SealedError):
        read_scores(seals, ledger, "r1", "fd")

    # unsealing requires DONE markers
    m1, m2 = tmp_path / "runs/npo/DONE", tmp_path / "runs/rmu/DONE"
    with pytest.raises(SealedError):
        unseal(seals, ledger, "r1", "fd", [m1, m2])
    for m in (m1, m2):
        m.parent.mkdir(parents=True)
        m.touch()
    assert unseal(seals, ledger, "r1", "fd", [m1, m2]) == SCORES
    assert read_scores(seals, ledger, "r1", "fd") == SCORES

    # ledger has both entries, in order
    lines = ledger.read_text().strip().splitlines()
    assert len(lines) == 2 and '"sealed"' in lines[0] and '"opened"' in lines[1]


def test_double_seal_raises(tmp_path):
    seals, ledger = tmp_path / "seals", tmp_path / "ledger.jsonl"
    seal_scores(seals, ledger, "r1", "fd", SCORES)
    with pytest.raises(SealedError):
        seal_scores(seals, ledger, "r1", "fd", SCORES)


def test_tamper_detection(tmp_path):
    seals, ledger = tmp_path / "seals", tmp_path / "ledger.jsonl"
    seal_scores(seals, ledger, "r1", "fd", SCORES)
    m = tmp_path / "DONE"
    m.touch()
    unseal(seals, ledger, "r1", "fd", [m])
    (seals / "r1" / "fd.json").write_text('{"c00": 999.0}')
    with pytest.raises(SealedError):
        read_scores(seals, ledger, "r1", "fd")


def test_unseal_unknown_raises(tmp_path):
    with pytest.raises(SealedError):
        unseal(tmp_path / "seals", tmp_path / "ledger.jsonl", "r1", "fd", [])
