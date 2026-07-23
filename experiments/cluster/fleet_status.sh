#!/usr/bin/env bash
set -euo pipefail

# One command, whole fleet: per-node GPU/worker snapshots + every queue,
# with assignment-mismatch warnings. Works from any node and any directory.
#
#   bash /group-volume/jieuns.shin/retain-susceptibility/experiments/cluster/fleet_status.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "${ROOT}/experiments/cluster/fleet_status.py"
