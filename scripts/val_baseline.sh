#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
WEIGHTS="${WEIGHTS:-${ROOT}/runs/repro_compare/baseline/weights/best.pt}"
exec "${PYTHON:-python}" scripts/eval_repro.py --weights "${WEIGHTS}" "$@"
