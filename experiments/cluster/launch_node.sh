#!/usr/bin/env bash
set -Eeuo pipefail

# Bring one 8xH100 node into the fleet: start one queue worker per GPU.
#
#   bash experiments/cluster/launch_node.sh <queue-dir> [n-gpus]
#
# Examples (repo lives on /group-volume, shared by every node):
#   bash experiments/cluster/launch_node.sh runs/cluster_queue/calib        # all visible GPUs
#   bash experiments/cluster/launch_node.sh runs/cluster_queue/calib 4     # GPUs 0-3 only
#   WAIT=0 bash experiments/cluster/launch_node.sh runs/cluster_queue/calib # exit when drained
#
# Workers are nohup'd, so the launcher returns immediately and survives the
# SSH session.  Stop a node's workers with:
#   pkill -f "experiments/cluster/worker.py --queue"

QUEUE="${1:-}"
if [[ -z "${QUEUE}" ]]; then
  echo "usage: $0 <queue-dir> [n-gpus]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-/group-volume/jieuns.shin/venvs/exp}"
WAIT="${WAIT:-1}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "missing official environment: ${VENV}/bin/activate" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"
cd "${ROOT}"
export HF_HOME="${HF_HOME:-/group-volume/data/hf_home}"
export PYTHONUNBUFFERED=1

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

HOST="$(hostname)"
LOGDIR="runs/logs/cluster"
mkdir -p "${LOGDIR}"
python experiments/cluster/workqueue.py init --queue "${QUEUE}"

wait_flag=()
if [[ "${WAIT}" == "1" ]]; then
  wait_flag=(--wait)
fi

echo "node=${HOST} gpus=${NGPU}/${DETECTED} queue=${QUEUE} wait=${WAIT}"
for (( g = 0; g < NGPU; g++ )); do
  out="${LOGDIR}/worker_${HOST}_gpu${g}.out"
  nohup python -u experiments/cluster/worker.py \
    --queue "${QUEUE}" --gpu "${g}" "${wait_flag[@]}" \
    >> "${out}" 2>&1 &
  echo "  worker gpu${g} pid=$! log=${out}"
done
echo "monitor: python experiments/cluster/workqueue.py status --queue ${QUEUE}"
