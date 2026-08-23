#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="${PYTHON_BIN}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"
elif [[ -x "${PACKAGE_DIR}/.venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="${PACKAGE_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXECUTABLE="$(command -v python3)"
else
  echo "Python was not found. Activate the evaluation environment first." >&2
  exit 1
fi

if ! "${PYTHON_EXECUTABLE}" -c \
  "import torch, numpy, trimesh, scipy, smplx, yaml, easydict" \
  >/dev/null 2>&1; then
  echo "The selected Python is missing evaluation dependencies:" >&2
  echo "  ${PYTHON_EXECUTABLE}" >&2
  echo "Activate the text2hoi environment or run ${PACKAGE_DIR}/setup.sh." >&2
  exit 1
fi

echo "[GPU launcher] Python: ${PYTHON_EXECUTABLE}"
cd "${PACKAGE_DIR}"
exec "${PYTHON_EXECUTABLE}" \
  "${PACKAGE_DIR}/hot3d/visualize-code/eval_metric_gpu.py" "$@"
