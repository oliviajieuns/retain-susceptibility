#!/usr/bin/env bash
set -Eeuo pipefail

# Bring this node into the fleet: start its status watcher and one queue
# worker per GPU. Safe to re-run — GPUs that already have a worker for this
# queue are skipped, and the watcher is single-instance.
#
#   bash experiments/cluster/launch_node.sh              # queue from configs/cluster/fleet.yaml
#   bash experiments/cluster/launch_node.sh <queue-dir>  # explicit (must match assignment)
#   FORCE_QUEUE=1 bash ... <queue-dir>                   # override the assignment guard
#   WAIT=0 ...                                           # workers exit when queue drains
#
# Stop this node's workers:  pkill -f "experiments/cluster/worker.py --queue"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-/group-volume/jieuns.shin/venvs/exp}"
WAIT="${WAIT:-1}"
HOST="$(hostname)"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "missing official environment: ${VENV}/bin/activate" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
cd "${ROOT}"
export HF_HOME="${HF_HOME:-/group-volume/data/hf_home}"
export PYTHONUNBUFFERED=1

ASSIGNED="$(python - <<PY
import yaml, pathlib
cfg = pathlib.Path("configs/cluster/fleet.yaml")
data = yaml.safe_load(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}
print((data.get("assignments") or {}).get("${HOST}", ""))
PY
)"

QUEUE="${1:-}"
if [[ -z "${QUEUE}" ]]; then
  QUEUE="${ASSIGNED}"
  if [[ -z "${QUEUE}" ]]; then
    echo "node ${HOST} has no assignment in configs/cluster/fleet.yaml and no queue was given" >&2
    exit 2
  fi
elif [[ -n "${ASSIGNED}" && "${QUEUE%/}" != "${ASSIGNED%/}" && "${FORCE_QUEUE:-0}" != "1" ]]; then
  echo "refusing: ${HOST} is assigned to ${ASSIGNED}, not ${QUEUE}." >&2
  echo "Fix configs/cluster/fleet.yaml or rerun with FORCE_QUEUE=1." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found; this launcher is for GPU nodes" >&2
  exit 1
fi
DETECTED="$(nvidia-smi -L | wc -l)"
NGPU="${2:-${DETECTED}}"
if (( NGPU > DETECTED )); then
  echo "requested ${NGPU} GPUs but node has ${DETECTED}" >&2
  exit 1
fi

LOGDIR="runs/logs/cluster"
mkdir -p "${LOGDIR}"
python experiments/cluster/workqueue.py init --queue "${QUEUE}"

nohup python -u experiments/cluster/node_watch.py --replace \
  >> "${LOGDIR}/watch_${HOST}.out" 2>&1 &
echo "node=${HOST} queue=${QUEUE} gpus=${NGPU}/${DETECTED} wait=${WAIT} watcher_pid=$!"

wait_flag=()
if [[ "${WAIT}" == "1" ]]; then
  wait_flag=(--wait)
fi

for (( g = 0; g < NGPU; g++ )); do
  if pgrep -f "experiments/cluster/worker.py --queue ${QUEUE} --gpu ${g}( |$)" >/dev/null; then
    echo "  worker gpu${g}: already running, skipped"
    continue
  fi
  out="${LOGDIR}/worker_${HOST}_gpu${g}.out"
  nohup python -u experiments/cluster/worker.py \
    --queue "${QUEUE}" --gpu "${g}" "${wait_flag[@]}" \
    >> "${out}" 2>&1 &
  echo "  worker gpu${g}: started pid=$! log=${out}"
done
echo "overview: bash experiments/cluster/fleet_status.sh"
