#!/usr/bin/env bash
set -Eeuo pipefail

# Provision meta-llama/Llama-3.1-8B-Instruct into /group-volume/models for the
# llama8b_tofu.yaml campaign (second architecture family, Table 2).
#
# Run ON THE CLUSTER.  HF Hub has been blocked/unstable from the intranet since
# 2026-07-23 (CDN read timeouts), so the download is attempted in order:
#   1. plain huggingface-cli download (HF_HOME=/group-volume/data/hf_home)
#   2. retry through the mirror HF_ENDPOINT=https://hf-mirror.com
# and on double failure prints the local-download + rsync fallback.
#
# meta-llama is a GATED repo: an accepted license + a valid token are required.
# The script respects HUGGING_FACE_HUB_TOKEN / HF_TOKEN from the environment or
# a cached `huggingface-cli login`; it never embeds or asks for a token itself.
#
# Idempotent: if config.json and at least one *.safetensors shard are already
# present under the destination, it verifies and exits without downloading.

REPO_ID="meta-llama/Llama-3.1-8B-Instruct"
DEST="${DEST:-/group-volume/models/Llama-3.1-8B-Instruct}"
VENV="${VENV:-/group-volume/jieuns.shin/venvs/exp}"
export HF_HOME="${HF_HOME:-/group-volume/data/hf_home}"
# A stale offline flag would turn every attempt into an instant cache miss.
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE || true

log()  { echo "[provision_llama] $*"; }
fail() { echo "[provision_llama] ERROR: $*" >&2; exit 1; }

verify_dest() {
  # Returns 0 when the destination looks like a complete HF model snapshot.
  [[ -f "${DEST}/config.json" ]] || return 1
  local n_st
  n_st="$(find "${DEST}" -maxdepth 1 -name '*.safetensors' | wc -l)"
  (( n_st >= 1 )) || return 1
  # Llama-3.1-8B-Instruct ships 4 safetensors shards, ~15GiB total in bf16.
  local total_bytes
  total_bytes="$(find "${DEST}" -maxdepth 1 -name '*.safetensors' -printf '%s\n' \
    | awk '{s+=$1} END {print s+0}')"
  log "found ${n_st} safetensors shard(s), $(( total_bytes / 1024 / 1024 / 1024 )) GiB at ${DEST}"
  if (( total_bytes < 14 * 1024 * 1024 * 1024 )); then
    log "WARNING: total shard size below the expected ~15GiB — download may be truncated."
    return 1
  fi
  [[ -f "${DEST}/tokenizer.json" || -f "${DEST}/tokenizer.model" ]] || {
    log "WARNING: no tokenizer file next to the shards."
    return 1
  }
  return 0
}

report_gated_hint() {
  # $1 = log file of the failed attempt
  if grep -qiE '401|403|gated|access.*restricted|awaiting.*review|Cannot access gated' "$1"; then
    cat >&2 <<'EOF'
[provision_llama] The failure looks like a GATED-REPO auth error (401/403).
  meta-llama/Llama-3.1-8B-Instruct requires:
    1. an HF account that has accepted the Llama 3.1 license on the model page,
    2. a token visible to this shell: export HUGGING_FACE_HUB_TOKEN=hf_...
       (or `huggingface-cli login` once; the cached token under HF_HOME works).
  Fix the token/license first — the mirror cannot bypass gating either.
EOF
  fi
}

if verify_dest; then
  log "destination already complete — nothing to do."
  exit 0
fi

if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
fi
command -v huggingface-cli >/dev/null 2>&1 \
  || fail "huggingface-cli not found (activate ${VENV} or pip install -U huggingface_hub)"

if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}${HF_TOKEN:-}" ]]; then
  log "using HF token from environment."
elif [[ -f "${HF_HOME}/token" || -f "${HOME}/.cache/huggingface/token" ]]; then
  log "using cached huggingface-cli login token."
else
  log "WARNING: no HF token found in env or cache — the gated download will fail with 401."
fi

mkdir -p "${DEST}"
TMPLOG="$(mktemp /tmp/provision_llama.XXXXXX.log)"

download() {
  # $1 = attempt label; extra env comes from the caller
  log "attempt: $1"
  if huggingface-cli download "${REPO_ID}" \
      --local-dir "${DEST}" \
      --exclude "original/*" \
      2>&1 | tee "${TMPLOG}"; then
    return 0
  fi
  return 1
}

ok=0
# Attempt 1: plain hub.
if download "direct HF Hub"; then
  ok=1
else
  report_gated_hint "${TMPLOG}"
  log "direct download failed — retrying via hf-mirror.com (Hub blocked/unstable on the intranet since 2026-07-23)."
  # Attempt 2: mirror endpoint.
  if HF_ENDPOINT=https://hf-mirror.com download "hf-mirror.com"; then
    ok=1
  else
    report_gated_hint "${TMPLOG}"
  fi
fi

if (( ok == 0 )); then
  cat >&2 <<EOF
[provision_llama] Both download routes failed.
Fallback — download OUTSIDE the intranet, then bring the files in:

  # on a machine with HF access (license accepted, token set):
  huggingface-cli download ${REPO_ID} \\
      --local-dir ./Llama-3.1-8B-Instruct --exclude "original/*"
  # then transfer into the shared volume (rsync/scp per your ingress route):
  rsync -avP ./Llama-3.1-8B-Instruct/ <cluster>:${DEST}/

Re-run this script afterwards; it only verifies once the files are in place.
EOF
  exit 1
fi

log "download finished — verifying."
verify_dest || fail "verification failed after download; inspect ${DEST} and ${TMPLOG}"
log "OK: ${REPO_ID} provisioned at ${DEST}"
find "${DEST}" -maxdepth 1 -type f -printf '%10s  %f\n' | sort -k2
