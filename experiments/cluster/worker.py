"""Per-GPU queue worker.

One worker process owns one GPU for its whole lifetime and drains units from a
shared-volume queue (see ``workqueue.py``).  ``launch_node.sh`` starts one worker
per GPU on a node, so an 8-GPU node contributes eight concurrent workers and a
13-node fleet contributes ~104 without any further coordination.

The worker refuses to start on a GPU that already has resident memory unless
``--allow-busy-gpu`` is passed: fp32 7B runs need most of an H100 and a
double-booked GPU kills the run that got there first.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from workqueue import Claim, WorkQueue  # noqa: E402

HEARTBEAT_INTERVAL_S = 60.0


def gpu_memory_used_mib(gpu: int) -> int | None:
    """Resident memory on the GPU in MiB, or None when nvidia-smi is absent."""
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", str(gpu)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def build_env(base: dict[str, str], unit_env: dict[str, str], gpu: int, needs_gpu: bool) -> dict[str, str]:
    env = dict(base)
    env.update(unit_env)
    if needs_gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["GPU"] = str(gpu)  # h100_campaign.sh reads GPU=
    env.setdefault("PYTHONUNBUFFERED", "1")
    # HF Hub is blocked/unstable from the cluster (2026-07-23); every queued
    # unit must run cache-only unless its own env explicitly opts back in
    # (e.g. a provisioning unit setting HF_HUB_OFFLINE=0).
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def run_claim(queue: WorkQueue, claim: Claim, gpu: int, log_dir: Path) -> bool:
    unit = claim.unit
    host = socket.gethostname()
    attempt = claim.attempts + 1
    log_path = log_dir / f"{unit.unit_id}__{host}_gpu{gpu}__try{attempt}.out"
    env = build_env(os.environ.copy(), unit.env, gpu, needs_gpu=unit.gpus > 0)

    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(HEARTBEAT_INTERVAL_S):
            try:
                queue.heartbeat(unit.unit_id)
            except OSError:
                pass  # transient NFS hiccup; the next beat retries

    beater = threading.Thread(target=beat, daemon=True)
    beater.start()
    started = time.time()
    print(f"[worker gpu{gpu}] start {unit.unit_id} (attempt {attempt}) -> {log_path}", flush=True)
    exit_code: int | None = None
    error: str | None = None
    try:
        with open(log_path, "ab") as log:
            header = (
                f"# unit={unit.unit_id} attempt={attempt} host={host} gpu={gpu}\n"
                f"# cmd={' '.join(unit.cmd)}\n"
            )
            log.write(header.encode("utf-8"))
            log.flush()
            proc = subprocess.run(
                unit.cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        exit_code = proc.returncode
    except Exception as exc:  # a broken unit must never take the worker down
        error = f"{type(exc).__name__}: {exc}"
        try:
            with open(log_path, "ab") as log:
                log.write(f"# worker error: {error}\n".encode("utf-8"))
        except OSError:
            pass
    finally:
        stop.set()
        beater.join(timeout=5)

    result = {
        "exit_code": exit_code,
        "error": error,
        "host": host,
        "gpu": gpu,
        "duration_s": round(time.time() - started, 1),
        "log": str(log_path),
    }
    if exit_code == 0:
        queue.complete(claim, result)
        print(f"[worker gpu{gpu}] done {unit.unit_id} ({result['duration_s']}s)", flush=True)
        return True
    state = queue.fail(claim, result)
    print(
        f"[worker gpu{gpu}] FAIL {unit.unit_id} exit={exit_code} error={error} -> {state} "
        f"(log: {log_path})",
        flush=True,
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, help="queue root directory")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index this worker owns")
    parser.add_argument("--wait", action="store_true",
                        help="keep polling for new units instead of exiting on an empty queue")
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--busy-threshold-mib", type=int, default=1024)
    parser.add_argument("--log-dir", default=str(ROOT / "runs" / "logs" / "cluster"))
    args = parser.parse_args()

    used = gpu_memory_used_mib(args.gpu)
    if used is not None and used > args.busy_threshold_mib and not args.allow_busy_gpu:
        raise SystemExit(
            f"GPU {args.gpu} already has {used} MiB resident; refusing to double-book "
            "(pass --allow-busy-gpu to override)"
        )

    queue = WorkQueue(Path(args.queue))
    queue.init()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    owner = {"host": socket.gethostname(), "gpu": args.gpu, "pid": os.getpid()}

    interrupted = {"flag": False}

    def on_term(signum, frame):  # noqa: ANN001
        interrupted["flag"] = True

    signal.signal(signal.SIGTERM, on_term)

    while not interrupted["flag"]:
        try:
            claim = queue.claim(owner=owner)
        except OSError as exc:
            # Transient shared-volume error: back off and retry, never die.
            print(f"[worker gpu{args.gpu}] claim error, retrying: {exc}", flush=True)
            time.sleep(args.poll_s)
            continue
        if claim is None:
            if not args.wait:
                print(f"[worker gpu{args.gpu}] queue drained; exiting", flush=True)
                return
            time.sleep(args.poll_s)
            continue
        try:
            run_claim(queue, claim, args.gpu, log_dir)
        except Exception as exc:
            # The claim stays in claimed/ and requeue-stale will recover it;
            # the worker itself must survive to serve the next unit.
            print(f"[worker gpu{args.gpu}] unit {claim.unit.unit_id} bookkeeping error: {exc}",
                  flush=True)
            time.sleep(args.poll_s)
    print(f"[worker gpu{args.gpu}] terminated by signal", flush=True)


if __name__ == "__main__":
    main()
