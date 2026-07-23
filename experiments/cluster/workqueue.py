"""Shared-filesystem work queue for multi-node H100 campaigns.

Every node mounts the same repository under /group-volume, so a directory of
JSON files is the coordination substrate: no scheduler, no daemon, no extra
dependency.  A unit moves through ``pending -> claimed -> done|failed`` by
atomic rename.  Workers on any node claim units concurrently; a claim that
loses the rename race simply moves on to the next pending unit.

Layout under ``--queue <root>``::

    pending/<unit_id>.json      enqueued unit + attempt counter
    claimed/<unit_id>.json      unit currently owned by a worker
    claimed/<unit_id>.meta.json owner token (host/gpu/pid/started)
    claimed/<unit_id>.hb        heartbeat file (mtime refreshed by the owner)
    done/<unit_id>.json         unit + result of the successful attempt
    failed/<unit_id>.json       unit + result after max_attempts exhausted

Crash recovery: a worker that dies stops refreshing its heartbeat.
``workqueue.py requeue-stale`` moves such units back to ``pending``.  Requeue can
in principle double-run a unit that is still alive but silent, so every
enqueued command must be resume-safe (all campaign runners here take
``--resume``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterable

STATES = ("pending", "claimed", "done", "failed")


@dataclasses.dataclass
class Unit:
    unit_id: str
    cmd: list[str]
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    gpus: int = 1
    max_attempts: int = 2

    def to_payload(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_payload(payload: dict) -> "Unit":
        return Unit(
            unit_id=str(payload["unit_id"]),
            cmd=[str(part) for part in payload["cmd"]],
            env={str(k): str(v) for k, v in payload.get("env", {}).items()},
            gpus=int(payload.get("gpus", 1)),
            max_attempts=int(payload.get("max_attempts", 2)),
        )


@dataclasses.dataclass
class Claim:
    unit: Unit
    attempts: int  # attempts already consumed before this one
    token: str
    path: Path  # claimed/<unit_id>.json


def _validate_unit_id(unit_id: str) -> None:
    ok = unit_id and all(ch.isalnum() or ch in "._-" for ch in unit_id)
    if not ok:
        raise ValueError(f"unit_id must be filesystem-safe [A-Za-z0-9._-]: {unit_id!r}")


class WorkQueue:
    def __init__(self, root: Path):
        self.root = Path(root)

    def init(self) -> None:
        for state in STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)

    def _state_dir(self, state: str) -> Path:
        return self.root / state

    def _entry(self, state: str, unit_id: str) -> Path:
        return self._state_dir(state) / f"{unit_id}.json"

    def _write_json(self, path: Path, payload: dict) -> None:
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _locate(self, unit_id: str) -> str | None:
        for state in STATES:
            if self._entry(state, unit_id).exists():
                return state
        return None

    # -- producer side ------------------------------------------------------

    def enqueue(self, units: Iterable[Unit]) -> list[str]:
        self.init()
        added = []
        for unit in units:
            _validate_unit_id(unit.unit_id)
            state = self._locate(unit.unit_id)
            if state is not None:
                raise FileExistsError(
                    f"unit {unit.unit_id} already exists in state {state!r}; "
                    "queues are append-only per unit_id — pick a new id"
                )
            self._write_json(
                self._entry("pending", unit.unit_id),
                {"unit": unit.to_payload(), "attempts": 0},
            )
            added.append(unit.unit_id)
        return added

    # -- worker side --------------------------------------------------------

    def claim(self, owner: dict | None = None) -> Claim | None:
        """Claim one pending unit, or return None when nothing is claimable."""
        pending = sorted(self._state_dir("pending").glob("*.json"))
        token = uuid.uuid4().hex
        for path in pending:
            dst = self._entry("claimed", path.stem)
            try:
                os.replace(path, dst)
            except FileNotFoundError:
                continue  # lost the race for this unit
            # NFS can duplicate a rename reply; the owner token written after
            # the rename disambiguates: last writer owns, the loser re-reads
            # and walks away without touching the unit.
            meta = dict(owner or {})
            meta.update({"token": token, "claimed_at": time.time()})
            meta_path = dst.with_name(f"{path.stem}.meta.json")
            self._write_json(meta_path, meta)
            observed = json.loads(meta_path.read_text(encoding="utf-8"))
            if observed.get("token") != token:
                continue
            self.heartbeat(path.stem)
            payload = json.loads(dst.read_text(encoding="utf-8"))
            return Claim(
                unit=Unit.from_payload(payload["unit"]),
                attempts=int(payload.get("attempts", 0)),
                token=token,
                path=dst,
            )
        return None

    def heartbeat(self, unit_id: str) -> None:
        hb = self._state_dir("claimed") / f"{unit_id}.hb"
        hb.touch()

    def _clear_claim(self, unit_id: str) -> None:
        for suffix in (".json", ".meta.json", ".hb"):
            path = self._state_dir("claimed") / f"{unit_id}{suffix}"
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def complete(self, claim: Claim, result: dict) -> None:
        self._finish(claim, "done", result)

    def fail(self, claim: Claim, result: dict) -> str:
        """Record a failed attempt.  Returns the resulting state."""
        attempts = claim.attempts + 1
        if attempts < claim.unit.max_attempts:
            self._write_json(
                self._entry("pending", claim.unit.unit_id),
                {"unit": claim.unit.to_payload(), "attempts": attempts,
                 "last_failure": result},
            )
            self._clear_claim(claim.unit.unit_id)
            return "pending"
        self._finish(claim, "failed", result, attempts=attempts)
        return "failed"

    def _finish(self, claim: Claim, state: str, result: dict, attempts: int | None = None) -> None:
        self._write_json(
            self._entry(state, claim.unit.unit_id),
            {
                "unit": claim.unit.to_payload(),
                "attempts": claim.attempts + 1 if attempts is None else attempts,
                "result": result,
            },
        )
        self._clear_claim(claim.unit.unit_id)

    # -- maintenance --------------------------------------------------------

    def requeue_stale(self, max_age_s: float, now: float | None = None) -> list[str]:
        """Return claimed units whose heartbeat is older than max_age_s to pending."""
        now = time.time() if now is None else now
        requeued = []
        for path in sorted(self._state_dir("claimed").glob("*.json")):
            if path.name.endswith(".meta.json"):
                continue
            unit_id = path.stem
            hb = self._state_dir("claimed") / f"{unit_id}.hb"
            stamp = hb if hb.exists() else path
            if now - stamp.stat().st_mtime <= max_age_s:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._write_json(
                self._entry("pending", unit_id),
                {"unit": payload["unit"],
                 "attempts": int(payload.get("attempts", 0)) + 1,
                 "last_failure": {"reason": f"stale heartbeat > {max_age_s}s"}},
            )
            self._clear_claim(unit_id)
            requeued.append(unit_id)
        return requeued

    def retry_failed(self) -> list[str]:
        """Move every failed unit back to pending with a fresh attempt budget."""
        retried = []
        for path in sorted(self._state_dir("failed").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self._write_json(
                self._entry("pending", path.stem),
                {"unit": payload["unit"], "attempts": 0,
                 "last_failure": payload.get("result", {})},
            )
            path.unlink()
            retried.append(path.stem)
        return retried

    def status(self) -> dict:
        report: dict = {"root": str(self.root), "counts": {}, "claimed": [], "failed": []}
        for state in STATES:
            entries = [
                p for p in self._state_dir(state).glob("*.json")
                if not p.name.endswith(".meta.json")
            ]
            report["counts"][state] = len(entries)
            if state == "claimed":
                now = time.time()
                for path in sorted(entries):
                    meta_path = path.with_name(f"{path.stem}.meta.json")
                    meta = (
                        json.loads(meta_path.read_text(encoding="utf-8"))
                        if meta_path.exists() else {}
                    )
                    hb = self._state_dir("claimed") / f"{path.stem}.hb"
                    age = now - (hb.stat().st_mtime if hb.exists() else path.stat().st_mtime)
                    report["claimed"].append({
                        "unit_id": path.stem,
                        "host": meta.get("host"),
                        "gpu": meta.get("gpu"),
                        "heartbeat_age_s": round(age, 1),
                    })
            if state == "failed":
                for path in sorted(entries):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    report["failed"].append({
                        "unit_id": path.stem,
                        "exit_code": payload.get("result", {}).get("exit_code"),
                        "log": payload.get("result", {}).get("log"),
                    })
        return report


def read_units_jsonl(path: Path) -> list[Unit]:
    units = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            units.append(Unit.from_payload(json.loads(line)))
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "enqueue", "status", "requeue-stale", "retry-failed"])
    parser.add_argument("--queue", required=True, help="queue root directory (on the shared volume)")
    parser.add_argument("--units", help="units JSONL file (enqueue)")
    parser.add_argument("--stale-after", type=float, default=1800.0,
                        help="requeue-stale: heartbeat age threshold in seconds")
    parser.add_argument("--brief", action="store_true",
                        help="status: one line per queue state instead of full JSON")
    args = parser.parse_args()

    queue = WorkQueue(Path(args.queue))
    if args.action == "init":
        queue.init()
        print(f"initialized {queue.root}")
    elif args.action == "enqueue":
        if not args.units:
            parser.error("enqueue requires --units <file.jsonl>")
        added = queue.enqueue(read_units_jsonl(Path(args.units)))
        print(f"enqueued {len(added)} unit(s)")
        for unit_id in added:
            print(f"  {unit_id}")
    elif args.action == "status":
        report = queue.status()
        if not args.brief:
            print(json.dumps(report, indent=2))
        counts = report["counts"]
        total = sum(counts.values())
        done = counts.get("done", 0)
        print(f"progress: {done}/{total} done, {counts.get('claimed', 0)} running, "
              f"{counts.get('pending', 0)} pending, {counts.get('failed', 0)} failed")
        if args.brief:
            for row in report["claimed"]:
                print(f"  RUN  {row['unit_id']}  {row['host']} gpu{row['gpu']} "
                      f"hb={row['heartbeat_age_s']}s")
            for row in report["failed"]:
                print(f"  FAIL {row['unit_id']}  exit={row['exit_code']}  {row['log']}")
    elif args.action == "requeue-stale":
        requeued = queue.requeue_stale(args.stale_after)
        print(f"requeued {len(requeued)} stale unit(s): {requeued}")
    elif args.action == "retry-failed":
        retried = queue.retry_failed()
        print(f"retried {len(retried)} failed unit(s): {retried}")


if __name__ == "__main__":
    main()
