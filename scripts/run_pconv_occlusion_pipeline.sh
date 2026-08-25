#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PIPELINE_PYTHON="${PIPELINE_PYTHON:-python}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-${REPO_ROOT}/configs/pconv_occlusion_compare.local.yaml}"
ORCHESTRATOR="${REPO_ROOT}/scripts/run_pconv_occlusion_compare.py"

if [[ ! -x "$(command -v "${PIPELINE_PYTHON}" 2>/dev/null)" && ! -x "${PIPELINE_PYTHON}" ]]; then
    echo "Python interpreter is not executable: ${PIPELINE_PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${PIPELINE_CONFIG}" ]]; then
    echo "Experiment config does not exist: ${PIPELINE_CONFIG}" >&2
    echo "Copy configs/pconv_occlusion_compare.example.yaml to the local config path and edit dataset paths." >&2
    exit 2
fi
if [[ ! -f "${ORCHESTRATOR}" ]]; then
    echo "Experiment orchestrator does not exist: ${ORCHESTRATOR}" >&2
    exit 2
fi

on_error() {
    local exit_code=$?
    echo "[$(date --iso-8601=seconds)] Pipeline failed with exit code ${exit_code}." >&2
    exit "${exit_code}"
}
trap on_error ERR

run_phase() {
    local phase=$1
    echo "[$(date --iso-8601=seconds)] Starting phase: ${phase}"
    "${PIPELINE_PYTHON}" -u "${ORCHESTRATOR}" "${phase}" --config "${PIPELINE_CONFIG}"
    echo "[$(date --iso-8601=seconds)] Completed phase: ${phase}"
}

cd "${REPO_ROOT}"
echo "[$(date --iso-8601=seconds)] PConv occlusion pipeline started."
echo "repository=${REPO_ROOT}"
echo "python=${PIPELINE_PYTHON}"
echo "config=${PIPELINE_CONFIG}"

# train-base resumes from last.pt when an incomplete 100-epoch run exists.
run_phase prepare
run_phase train-base
run_phase eval-base
run_phase finetune
run_phase eval-final

echo "[$(date --iso-8601=seconds)] PConv occlusion pipeline completed successfully."
