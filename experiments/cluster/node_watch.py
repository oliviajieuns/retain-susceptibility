"""Per-node status recorder for the fleet dashboard.

Runs on each node (started automatically by ``launch_node.sh``) and writes a
snapshot of the node's GPUs and queue workers to
``runs/cluster_status/<host>.json`` on the shared volume every ``--interval``
seconds, so ``fleet_status`` can show every node from any terminal.

Single-instance per host via a pidfile; a second launch exits quietly unless
``--replace`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def gather_gpus() -> list[dict]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        index, mem, util = [part.strip() for part in line.split(",")]
        gpus.append({"index": int(index), "mem_mib": int(mem), "util_pct": int(util)})
    return gpus


def parse_worker_cmdline(parts: list[str]) -> dict | None:
    """Extract {queue, gpu} from a worker.py command line, else None."""
    if not any(part.endswith("cluster/worker.py") or part.endswith("worker.py")
               for part in parts):
        return None
    if "--queue" not in parts or "--gpu" not in parts:
        return None
    try:
        return {
            "queue": parts[parts.index("--queue") + 1],
            "gpu": int(parts[parts.index("--gpu") + 1]),
        }
    except (IndexError, ValueError):
        return None


def gather_workers() -> list[dict]:
    workers = []
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            parts = (pid_dir / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        info = parse_worker_cmdline([p for p in parts if p])
        if info:
            info["pid"] = int(pid_dir.name)
            workers.append(info)
    return sorted(workers, key=lambda w: w["gpu"])


def write_snapshot(status_dir: Path, host: str) -> Path:
    status_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "host": host,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_epoch": time.time(),
        "gpus": gather_gpus(),
        "workers": gather_workers(),
        "watcher_pid": os.getpid(),
    }
    path = status_dir / f"{host}.json"
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--status-dir", default=str(ROOT / "runs" / "cluster_status"))
    parser.add_argument("--replace", action="store_true",
                        help="take over from an existing watcher on this host")
    parser.add_argument("--once", action="store_true", help="write one snapshot and exit")
    args = parser.parse_args()

    host = socket.gethostname()
    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    pidfile = status_dir / f"{host}.watcher.pid"

    if pidfile.exists():
        try:
            old = int(pidfile.read_text().strip())
        except ValueError:
            old = 0
        if old and old != os.getpid() and pid_alive(old):
            if not args.replace:
                print(f"watcher already running on {host} (pid {old}); exiting")
                return
            try:
                os.kill(old, 15)
            except OSError:
                pass
    pidfile.write_text(str(os.getpid()), encoding="utf-8")

    while True:
        try:
            write_snapshot(status_dir, host)
        except OSError as exc:
            print(f"snapshot failed (will retry): {exc}", file=sys.stderr)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
