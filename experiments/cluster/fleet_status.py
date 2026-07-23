"""Whole-fleet dashboard: per-node snapshots + every queue, with assignment checks.

Reads node snapshots written by ``node_watch.py`` (runs/cluster_status/) and
every queue under runs/cluster_queue/, and flags any worker or claimed unit
that contradicts configs/cluster/fleet.yaml. Works from any node and any
working directory.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from workqueue import WorkQueue  # noqa: E402


def load_assignments(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(host): str(queue) for host, queue in (data.get("assignments") or {}).items()}


def queue_key(queue_path: str | Path) -> str:
    """Normalize a queue reference to 'runs/cluster_queue/<name>' when possible."""
    path = Path(queue_path)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return str(path)
    return str(path).rstrip("/")


def assignment_mismatch(assignments: dict[str, str], host: str | None,
                        queue_path: str | Path) -> bool:
    """True when host has an assignment and this queue is not it."""
    if not host or host not in assignments:
        return False
    return queue_key(assignments[host]) != queue_key(queue_path)


def fmt_age(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 120 else f"{seconds / 60:.0f}m"


def main() -> None:
    assignments = load_assignments(ROOT / "configs" / "cluster" / "fleet.yaml")
    now = time.time()

    print("========== NODES ==========")
    status_dir = ROOT / "runs" / "cluster_status"
    snapshots = sorted(status_dir.glob("*.json")) if status_dir.exists() else []
    if not snapshots:
        print("(no node snapshots yet — launch_node.sh starts the watcher)")
    for snap_path in snapshots:
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        host = snap.get("host", snap_path.stem)
        age = now - float(snap.get("updated_epoch", 0))
        assigned = assignments.get(host, "(unassigned)")
        stale = "  ⚠ STALE snapshot" if age > 180 else ""
        print(f"{host}  [{assigned}]  updated {fmt_age(age)} ago{stale}")
        busy = {w["gpu"]: w for w in snap.get("workers", [])}
        for gpu in snap.get("gpus", []):
            worker = busy.get(gpu["index"])
            wtxt = ""
            if worker:
                wtxt = f"  worker->{queue_key(worker['queue'])}"
                if assignment_mismatch(assignments, host, worker["queue"]):
                    wtxt += "  ⚠ NOT ASSIGNED"
            print(f"  gpu{gpu['index']}: {gpu['mem_mib'] / 1024:.1f}GB {gpu['util_pct']}%{wtxt}")
        if not snap.get("gpus"):
            print("  (no GPUs reported)")

    print()
    print("========== QUEUES ==========")
    queue_root = ROOT / "runs" / "cluster_queue"
    queues = sorted(p for p in queue_root.iterdir() if p.is_dir()) if queue_root.exists() else []
    if not queues:
        print("(no queues)")
    for queue_path in queues:
        report = WorkQueue(queue_path).status()
        counts = report["counts"]
        total = sum(counts.values())
        owner = [h for h, q in assignments.items() if queue_key(q) == queue_key(queue_path)]
        owner_txt = f"  [node: {', '.join(owner)}]" if owner else ""
        print(f"{queue_path.name}{owner_txt}: {counts.get('done', 0)}/{total} done, "
              f"{counts.get('claimed', 0)} running, {counts.get('pending', 0)} pending, "
              f"{counts.get('failed', 0)} failed")
        for row in report["claimed"]:
            flag = "  ⚠ NOT ASSIGNED" if assignment_mismatch(
                assignments, row.get("host"), queue_path) else ""
            print(f"  RUN  {row['unit_id']}  {row.get('host')} gpu{row.get('gpu')} "
                  f"hb={row['heartbeat_age_s']}s{flag}")
        for row in report["failed"]:
            print(f"  FAIL {row['unit_id']}  exit={row.get('exit_code')}  {row.get('log')}")


if __name__ == "__main__":
    main()
