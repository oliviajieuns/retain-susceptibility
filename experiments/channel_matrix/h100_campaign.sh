#!/usr/bin/env bash
set -Eeuo pipefail

# Safe H100 entry point for the sealed 7B/8B channel-matrix campaign.
#
# Examples:
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh preflight
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh fidelity
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh calibration
#   bash experiments/channel_matrix/h100_campaign.sh select-freeze
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh audit
#   bash experiments/channel_matrix/h100_campaign.sh aggregate
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh alpha-development
#   bash experiments/channel_matrix/h100_campaign.sh select-alpha-freeze
#   GPU=0 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh alpha-audit
#   bash experiments/channel_matrix/h100_campaign.sh alpha-aggregate
# Two-GPU request sharding (use disjoint AUTHORS values):
#   GPU=0 AUTHORS=198 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh calibration
#   GPU=1 AUTHORS=199 MODEL_ID=qwen25_7b bash experiments/channel_matrix/h100_campaign.sh calibration

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-/group-volume/jieuns.shin/venvs/exp}"
CONFIG="${CONFIG:-configs/channel_matrix/7b_tofu.yaml}"
MODEL_ID="${MODEL_ID:-qwen25_7b}"
GPU="${GPU:-0}"
AUTHORS="${AUTHORS:-}"
ACTION="${1:-}"

if [[ -z "${ACTION}" ]]; then
  echo "usage: GPU=<index> MODEL_ID=<alias|all> $0 {preflight|prefetch|dry-calibration|fidelity|calibration|select-freeze|audit|aggregate|dry-alpha-development|alpha-development|select-alpha-freeze|dry-alpha-audit|alpha-audit|alpha-aggregate}" >&2
  exit 2
fi

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "missing official environment: ${VENV}/bin/activate" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
cd "${ROOT}"
export HF_HOME="${HF_HOME:-/group-volume/data/hf_home}"
export PYTHONUNBUFFERED=1
mkdir -p runs/logs

model_args=()
if [[ "${MODEL_ID}" != "all" ]]; then
  model_args=(--model-id "${MODEL_ID}")
fi

author_args=()
if [[ -n "${AUTHORS}" ]]; then
  author_args=(--only-authors "${AUTHORS}")
fi

preflight() {
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "HF_HOME=${HF_HOME}"
  echo "CUDA_VISIBLE_DEVICES=${GPU}"
  echo "AUTHORS=${AUTHORS:-all-phase-authors}"
  git status --short
  nvidia-smi
  CUDA_VISIBLE_DEVICES="${GPU}" python - "${CONFIG}" "${MODEL_ID}" <<'PY'
from pathlib import Path
import sys

import datasets
import sentence_transformers
import torch
import transformers
import yaml

config_path, selected = sys.argv[1:]
cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
models = [model for model in cfg["models"] if model.get("enabled", True)]
if selected != "all":
    models = [model for model in models if model["id"] == selected]
if not models:
    raise SystemExit(f"no enabled model matches MODEL_ID={selected!r}")
missing = [f"{model['id']}={model['path']}" for model in models if not Path(model["path"]).is_dir()]
if missing:
    raise SystemExit("missing model path(s): " + ", ".join(missing))
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")
print(f"transformers={transformers.__version__} datasets={datasets.__version__}")
print(f"sentence-transformers={sentence_transformers.__version__}")
print("models=" + ",".join(model["id"] for model in models))
PY
}

prefetch() {
  # Audit itself is forced offline. Populate every non-model dependency before
  # the freeze and audit boundary.
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
  python - <<'PY'
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

for subset in ("full", "forget10_perturbed"):
    split = load_dataset("locuslab/TOFU", subset)["train"]
    print(f"cached locuslab/TOFU/{subset}: {len(split)} rows")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print("cached sentence-transformers/all-MiniLM-L6-v2")
PY
}

run_phase() {
  local phase="$1"
  local phase_author_args=()
  if [[ "${phase}" != "fidelity" ]]; then
    phase_author_args=("${author_args[@]}")
  elif [[ -n "${AUTHORS}" ]]; then
    echo "AUTHORS is not applicable to the single frozen fidelity cell" >&2
    return 2
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python -u experiments/channel_matrix/run_campaign.py \
    --config "${CONFIG}" \
    --phase "${phase}" \
    --resume \
    "${phase_author_args[@]}" \
    "${model_args[@]}"
}

run_alpha_phase() {
  local phase="$1"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u experiments/channel_matrix/alpha_protection.py \
    --config "${CONFIG}" \
    --phase "${phase}" \
    --resume \
    "${author_args[@]}" \
    "${model_args[@]}"
}

case "${ACTION}" in
  preflight)
    preflight
    ;;
  prefetch)
    prefetch
    ;;
  dry-calibration)
    CUDA_VISIBLE_DEVICES="${GPU}" python experiments/channel_matrix/run_campaign.py \
      --config "${CONFIG}" \
      --phase calibration \
      --dry-run \
      "${author_args[@]}" \
      "${model_args[@]}"
    ;;
  fidelity)
    preflight
    run_phase fidelity
    ;;
  calibration)
    preflight
    run_phase calibration
    ;;
  select-freeze)
    python experiments/channel_matrix/select_freeze.py \
      --config "${CONFIG}" \
      --root runs/channel_matrix_7b/calibration \
      --out runs/channel_matrix_7b/objective_freeze.recommended.yaml
    echo "STOP: review the recommendation and commit a frozen objective_freeze.yaml before audit."
    ;;
  audit)
    if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
      echo "refusing audit: git worktree is dirty" >&2
      git status --short >&2
      exit 1
    fi
    preflight
    run_phase audit
    ;;
  aggregate)
    python experiments/channel_matrix/aggregate.py \
      --root runs/channel_matrix_7b/audit \
      --out runs/channel_matrix_7b/aggregate \
      --n-boot 2000
    python experiments/channel_matrix/make_main_table.py \
      --report runs/channel_matrix_7b/aggregate/pooled_channel_report.csv \
      --summary runs/channel_matrix_7b/aggregate/pooled_channel_report.json \
      --out docs/tables/table1_channel_matrix_7b.tex \
      --stress-out docs/tables/table1_stress_7b.tex
    ;;
  dry-alpha-development)
    CUDA_VISIBLE_DEVICES="${GPU}" python experiments/channel_matrix/alpha_protection.py \
      --config "${CONFIG}" \
      --phase development \
      --dry-run \
      "${author_args[@]}" \
      "${model_args[@]}"
    ;;
  alpha-development)
    preflight
    run_alpha_phase development
    ;;
  select-alpha-freeze)
    python experiments/channel_matrix/select_alpha_freeze.py \
      --config "${CONFIG}" \
      --root runs/channel_matrix_7b/alpha_protection/development \
      --out runs/channel_matrix_7b/alpha_protection_freeze.recommended.yaml
    echo "STOP: review and commit configs/channel_matrix/alpha_protection_freeze.yaml before alpha audit."
    ;;
  dry-alpha-audit)
    CUDA_VISIBLE_DEVICES="${GPU}" python experiments/channel_matrix/alpha_protection.py \
      --config "${CONFIG}" \
      --phase audit \
      --dry-run \
      "${author_args[@]}" \
      "${model_args[@]}"
    ;;
  alpha-audit)
    if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
      echo "refusing alpha audit: git worktree is dirty" >&2
      git status --short >&2
      exit 1
    fi
    preflight
    run_alpha_phase audit
    ;;
  alpha-aggregate)
    python experiments/channel_matrix/aggregate_alpha_protection.py \
      --config "${CONFIG}" \
      --root runs/channel_matrix_7b/alpha_protection/audit \
      --out runs/channel_matrix_7b/alpha_protection/aggregate \
      --n-boot 2000
    ;;
  *)
    echo "unknown action: ${ACTION}" >&2
    exit 2
    ;;
esac
