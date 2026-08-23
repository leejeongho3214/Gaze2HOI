#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PACKAGE_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Virtual environment is missing. Run ${PACKAGE_DIR}/setup.sh first." >&2
  exit 1
fi

cd "${PACKAGE_DIR}"
exec "${PYTHON_BIN}" "${PACKAGE_DIR}/hot3d/visualize-code/eval_metric.py" "$@"

