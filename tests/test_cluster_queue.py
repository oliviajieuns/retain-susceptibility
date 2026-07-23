"""CPU-only tests for the shared-filesystem cluster queue."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "cluster"))

import make_units  # noqa: E402
import worker  # noqa: E402
from workqueue import Unit, WorkQueue  # noqa: E402


def _unit(unit_id: str, cmd: list[str] | None = None, **kw) -> Unit:
    return Unit(unit_id=unit_id, cmd=cmd or ["true"], **kw)


def test_enqueue_claim_complete_roundtrip(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a"), _unit("b")])
    assert q.status()["counts"] == {"pending": 2, "claimed": 0, "done": 0, "failed": 0}

    first = q.claim(owner={"host": "n1", "gpu": 0})
    assert first is not None and first.unit.unit_id == "a"
    second = q.claim(owner={"host": "n1", "gpu": 1})
    assert second is not None and second.unit.unit_id == "b"
    assert q.claim() is None

    q.complete(first, {"exit_code": 0})
    q.complete(second, {"exit_code": 0})
    counts = q.status()["counts"]
    assert counts["done"] == 2 and counts["claimed"] == 0


def test_enqueue_refuses_duplicate_unit_id_in_any_state(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a")])
    with pytest.raises(FileExistsError):
        q.enqueue([_unit("a")])
    claim = q.claim()
    q.complete(claim, {"exit_code": 0})
    with pytest.raises(FileExistsError):
        q.enqueue([_unit("a")])


def test_unit_id_must_be_filesystem_safe(tmp_path):
    q = WorkQueue(tmp_path / "q")
    with pytest.raises(ValueError):
        q.enqueue([_unit("bad/id")])


def test_fail_requeues_until_max_attempts(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a", max_attempts=2)])

    claim = q.claim()
    assert q.fail(claim, {"exit_code": 1}) == "pending"
    counts = q.status()["counts"]
    assert counts["pending"] == 1 and counts["failed"] == 0

    claim = q.claim()
    assert claim.attempts == 1
    assert q.fail(claim, {"exit_code": 1}) == "failed"
    report = q.status()
    assert report["counts"]["failed"] == 1
    assert report["failed"][0]["exit_code"] == 1


def test_retry_failed_restores_full_attempt_budget(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a", max_attempts=1)])
    q.fail(q.claim(), {"exit_code": 3})
    assert q.status()["counts"]["failed"] == 1
    assert q.retry_failed() == ["a"]
    claim = q.claim()
    assert claim is not None and claim.attempts == 0


def test_requeue_stale_by_heartbeat_age(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a"), _unit("b")])
    q.claim(owner={"host": "n1", "gpu": 0})
    q.claim(owner={"host": "n2", "gpu": 0})

    # Viewed from one hour in the future, only b has kept beating.
    future = time.time() + 3600
    os.utime(q.root / "claimed" / "b.hb", times=(future, future))
    requeued = q.requeue_stale(max_age_s=1800, now=future)
    assert requeued == ["a"]
    counts = q.status()["counts"]
    assert counts["pending"] == 1 and counts["claimed"] == 1
    payload = json.loads((q.root / "pending" / "a.json").read_text(encoding="utf-8"))
    assert payload["attempts"] == 1


def test_claim_survives_concurrent_double_rename_semantics(tmp_path):
    # The losing side of a claim race sees FileNotFoundError on rename and
    # must move on to the next unit rather than crash.
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a"), _unit("b")])
    stolen = q.claim()
    assert stolen.unit.unit_id == "a"
    nxt = q.claim()
    assert nxt is not None and nxt.unit.unit_id == "b"


def test_worker_env_isolates_one_gpu_per_unit():
    env = worker.build_env({"PATH": "/bin"}, {"MODEL_ID": "qwen25_7b"}, gpu=5, needs_gpu=True)
    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["GPU"] == "5"
    assert env["MODEL_ID"] == "qwen25_7b"
    cpu_env = worker.build_env({}, {}, gpu=5, needs_gpu=False)
    assert "CUDA_VISIBLE_DEVICES" not in cpu_env


def test_worker_executes_units_and_records_results(tmp_path):
    q = WorkQueue(tmp_path / "q")
    marker = tmp_path / "ran.txt"
    q.enqueue([
        Unit(unit_id="ok", cmd=["sh", "-c", f"echo hello > {marker}"], gpus=0),
        Unit(unit_id="boom", cmd=["sh", "-c", "exit 7"], gpus=0, max_attempts=1),
    ])
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    while (claim := q.claim(owner={"host": "test", "gpu": -1})) is not None:
        worker.run_claim(q, claim, gpu=-1, log_dir=log_dir)

    report = q.status()
    assert report["counts"]["done"] == 1 and report["counts"]["failed"] == 1
    assert marker.read_text().strip() == "hello"
    assert report["failed"][0]["exit_code"] == 7
    failed_log = Path(report["failed"][0]["log"])
    assert failed_log.exists() and "boom" in failed_log.name


def test_make_units_calibration_shards_by_model_and_author():
    cfg = yaml.safe_load(
        (ROOT / "configs/channel_matrix/7b_tofu.yaml").read_text(encoding="utf-8")
    )
    units = make_units.build_units(
        cfg, "configs/channel_matrix/7b_tofu.yaml", "calibration", ["qwen25_7b"], 2
    )
    assert [u.unit_id for u in units] == ["cal__qwen25_7b__a198", "cal__qwen25_7b__a199"]
    for u in units:
        assert "--only-authors" in u.cmd and "--resume" in u.cmd and u.gpus == 1


def test_make_units_alpha_phases_shard_by_author_and_seed():
    cfg = yaml.safe_load(
        (ROOT / "configs/channel_matrix/7b_tofu.yaml").read_text(encoding="utf-8")
    )
    dev = make_units.build_units(
        cfg, "configs/channel_matrix/7b_tofu.yaml", "alpha-development", ["qwen25_7b"], 2
    )
    audit = make_units.build_units(
        cfg, "configs/channel_matrix/7b_tofu.yaml", "alpha-audit", ["qwen25_7b"], 2
    )
    dev_authors = set(cfg["alpha_protection"]["development"]["authors"])
    audit_block = cfg["alpha_protection"]["audit"]
    assert len(dev) == len(dev_authors) * len(cfg["alpha_protection"]["development"]["seeds"])
    assert len(audit) == len(audit_block["authors"]) * len(audit_block["seeds"])
    for u in dev + audit:
        assert "--worker" in u.cmd and "--author" in u.cmd and "--seed" in u.cmd
    assert not {u.unit_id for u in dev} & {u.unit_id for u in audit}


def test_make_units_rejects_disabled_model():
    cfg = yaml.safe_load(
        (ROOT / "configs/channel_matrix/7b_tofu.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError):
        make_units._enabled_models(cfg, {"llama31_8b"})


def test_queue_cli_roundtrip(tmp_path):
    units_file = tmp_path / "units.jsonl"
    units_file.write_text(
        json.dumps({"unit_id": "cli-a", "cmd": ["true"], "gpus": 0}) + "\n",
        encoding="utf-8",
    )
    queue_dir = tmp_path / "q"
    script = ROOT / "experiments/cluster/workqueue.py"
    for action, extra in [
        ("init", []),
        ("enqueue", ["--units", str(units_file)]),
        ("status", []),
        ("requeue-stale", []),
    ]:
        out = subprocess.run(
            [sys.executable, str(script), action, "--queue", str(queue_dir), *extra],
            capture_output=True, text=True,
        )
        assert out.returncode == 0, out.stderr
    assert (queue_dir / "pending" / "cli-a.json").exists()
