#!/usr/bin/env bash
set -euo pipefail

# One command, whole fleet: shows every queue under runs/cluster_queue.
# Works from any node and any working directory (shared volume).
#
#   bash /group-volume/jieuns.shin/retain-susceptibility/experiments/cluster/fleet_status.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
shopt -s nullglob
queues=("${ROOT}"/runs/cluster_queue/*/)
if (( ${#queues[@]} == 0 )); then
  echo "no queues under ${ROOT}/runs/cluster_queue"
  exit 0
fi
for q in "${queues[@]}"; do
  echo "===== $(basename "$q") ====="
  python "${ROOT}/experiments/cluster/workqueue.py" status --brief --queue "$q"
done
