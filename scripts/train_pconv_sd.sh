#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
exec "${PYTHON:-python}" scripts/train_repro.py --variant pconv_sdiou "$@"
