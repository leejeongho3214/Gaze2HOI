#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${PACKAGE_DIR}/.venv"
"${PACKAGE_DIR}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"${PACKAGE_DIR}/.venv/bin/python" -m pip install --no-build-isolation \
  -r "${PACKAGE_DIR}/requirements.txt"

echo "Setup complete. Run: ${PACKAGE_DIR}/run_eval.sh"
