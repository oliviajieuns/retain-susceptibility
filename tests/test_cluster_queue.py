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
        Unit(unit_id="ok", cmd=["sh", "-c", f"echo hello > {marker.as_posix()}"], gpus=0),
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


def test_worker_survives_unlaunchable_command(tmp_path):
    # A typo'd custom unit (missing binary) must be recorded as a failure,
    # not crash the worker and orphan the claim.
    q = WorkQueue(tmp_path / "q")
    q.enqueue([
        Unit(unit_id="typo", cmd=["/no/such/binary"], gpus=0, max_attempts=1),
        Unit(unit_id="after", cmd=["true"], gpus=0),
    ])
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    while (claim := q.claim(owner={"host": "test", "gpu": -1})) is not None:
        worker.run_claim(q, claim, gpu=-1, log_dir=log_dir)

    report = q.status()
    assert report["counts"] == {"pending": 0, "claimed": 0, "done": 1, "failed": 1}
    failed = json.loads((q.root / "failed" / "typo.json").read_text(encoding="utf-8"))
    assert failed["result"]["exit_code"] is None
    assert "FileNotFoundError" in failed["result"]["error"]


def test_make_units_calibration_shards_by_author_and_objective():
    cfg = yaml.safe_load(
        (ROOT / "configs/channel_matrix/7b_tofu.yaml").read_text(encoding="utf-8")
    )
    units = make_units.build_units(
        cfg, "configs/channel_matrix/7b_tofu.yaml", "calibration", ["qwen25_7b"], 2
    )
    objectives = list(cfg["calibration"]["objective_grid"])
    authors = cfg["calibration"]["authors"]
    assert len(units) == len(authors) * len(objectives)
    assert units[0].unit_id == f"cal__qwen25_7b__a{authors[0]}__{objectives[0]}"
    for u in units:
        assert "--only-authors" in u.cmd and "--only-objectives" in u.cmd
        assert "--resume" in u.cmd and u.gpus == 1
        assert u.cmd[u.cmd.index("--only-objectives") + 1] in objectives

    audit_units = make_units.build_units(
        cfg, "configs/channel_matrix/7b_tofu.yaml", "audit", ["qwen25_7b"], 2
    )
    assert [u.unit_id for u in audit_units] == [
        f"aud__qwen25_7b__a{a}" for a in cfg["audit"]["authors"]
    ]
    for u in audit_units:
        assert "--only-objectives" not in u.cmd


def test_cancel_moves_pending_and_claimed_units_to_failed(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("a"), _unit("b")])
    q.claim(owner={"host": "n1", "gpu": 0})  # claims a
    assert q.cancel("a") == "claimed"
    assert q.cancel("b") == "pending"
    report = q.status()
    assert report["counts"] == {"pending": 0, "claimed": 0, "done": 0, "failed": 2}
    with pytest.raises(FileNotFoundError):
        q.cancel("a")
    # cancelled units can be revived explicitly
    assert sorted(q.retry_failed()) == ["a", "b"]


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


def test_replicate_units_rebuild_exact_cli_with_seed_override():
    import make_replicate_units as mru

    gate_source = (ROOT / "experiments/gate_1p5b/gate.py").read_text(encoding="utf-8")
    flags = mru.parse_gate_flags(gate_source)
    # spot-check the parser map against known gate.py flags
    assert flags["seed"]["flag"] == "--seed" and not flags["seed"]["store_true"]
    assert flags["require_sft_target"]["store_true"]
    assert flags["probe_dirs"]["flag"] == "--probe-dirs"

    manifest = {
        "model_id": "qwen25_1p5b",
        "objectives": ["npo"],
        "cli": {
            "model": "/group-volume/models/Qwen2.5-1.5B-Instruct",
            "seed": 2025,
            "probe_dirs": 64,
            "gen_lr": 2e-06,
            "require_sft_target": True,
            "smoke": False,
            "run_tag": "chanbal2",
            "out_dir": "",
            "extra_predictors": "fd_norm",
        },
    }
    units = mru.build_units(manifest, [2026, 2027], "chanbal2", "python", gate_source)
    assert [u.unit_id for u in units] == ["gate__chanbal2-s2026", "gate__chanbal2-s2027"]
    for unit, seed in zip(units, [2026, 2027]):
        cmd = unit.cmd
        assert unit.max_attempts == 1  # gate run tags are append-only
        assert cmd[cmd.index("--seed") + 1] == str(seed)
        assert cmd[cmd.index("--run-tag") + 1] == f"chanbal2-s{seed}"
        assert "--out-dir" not in cmd and "--smoke" not in cmd
        assert "--require-sft-target" in cmd
        assert cmd[cmd.index("--extra-predictors") + 1] == "fd_norm"
        assert cmd[cmd.index("--gen-lr") + 1] == "2e-06"

    # replicating the source seed itself is an error
    with pytest.raises(ValueError):
        mru.build_units(manifest, [2025], "chanbal2", "python", gate_source)

    # CLI drift (manifest key unknown to today's gate.py) must be a hard error
    drifted = {"model_id": "m", "objectives": [], "cli": {"seed": 2025, "removed_flag": 1}}
    with pytest.raises(ValueError, match="drifted"):
        mru.build_units(drifted, [2026], "t", "python", gate_source)


def test_replicate_seed_expansion():
    import make_replicate_units as mru

    assert mru.expand_seeds("2026-2029,2040") == [2026, 2027, 2028, 2029, 2040]
    with pytest.raises(ValueError):
        mru.expand_seeds("2026,2026")


def test_fleet_assignment_mismatch_detection(tmp_path):
    import fleet_status as fs

    cfg = tmp_path / "fleet.yaml"
    cfg.write_text(
        "assignments:\n"
        "  node-a: runs/cluster_queue/wave1\n"
        "  node-b: runs/cluster_queue/wave1_14b\n",
        encoding="utf-8",
    )
    assignments = fs.load_assignments(cfg)
    assert assignments == {"node-a": "runs/cluster_queue/wave1",
                           "node-b": "runs/cluster_queue/wave1_14b"}
    # node serving its own queue: fine (relative or absolute path spelling)
    assert not fs.assignment_mismatch(assignments, "node-a", "runs/cluster_queue/wave1")
    assert not fs.assignment_mismatch(
        assignments, "node-a", fs.ROOT / "runs/cluster_queue/wave1")
    # node grabbing another campaign's queue: flagged
    assert fs.assignment_mismatch(assignments, "node-b", "runs/cluster_queue/wave1")
    # unknown host: never flagged
    assert not fs.assignment_mismatch(assignments, "node-c", "runs/cluster_queue/wave1")
    assert fs.load_assignments(tmp_path / "missing.yaml") == {}


def test_node_watch_snapshot_and_worker_parsing(tmp_path):
    import node_watch as nw

    parsed = nw.parse_worker_cmdline(
        ["python", "-u", "experiments/cluster/worker.py",
         "--queue", "runs/cluster_queue/wave1", "--gpu", "3", "--wait"])
    assert parsed == {"queue": "runs/cluster_queue/wave1", "gpu": 3}
    assert nw.parse_worker_cmdline(["python", "train.py", "--gpu", "3"]) is None
    assert nw.parse_worker_cmdline(
        ["python", "experiments/cluster/worker.py", "--gpu", "3"]) is None

    path = nw.write_snapshot(tmp_path / "status", "test-host")
    snap = json.loads(path.read_text(encoding="utf-8"))
    assert snap["host"] == "test-host"
    assert isinstance(snap["gpus"], list) and isinstance(snap["workers"], list)
    assert snap["updated_epoch"] > 0


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


def test_enqueue_skip_existing_tops_up_after_duplicates(tmp_path):
    q = WorkQueue(tmp_path / "q")
    q.enqueue([_unit("u0"), _unit("u1")])
    # Default stays append-only strict.
    try:
        q.enqueue([_unit("u1"), _unit("u2")])
    except FileExistsError:
        pass
    else:  # pragma: no cover
        raise AssertionError("duplicate must raise without skip_existing")
    # skip_existing adds the units AFTER the duplicate instead of aborting.
    added = q.enqueue([_unit("u1"), _unit("u3")], skip_existing=True)
    assert added == ["u3"]
    assert q.last_skipped == ["u1"]
    # u0 u1 u3 — the strict call aborted at duplicate u1, losing u2 entirely;
    # that lost-tail behavior is exactly why top-ups must pass skip_existing.
    assert q.status()["counts"]["pending"] == 3
