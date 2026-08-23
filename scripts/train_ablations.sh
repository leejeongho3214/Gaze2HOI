#!/usr/bin/env bash
# Retrain every Table 2 / Table 3 ablation at a common 100k-iteration budget.
#
#   RUN_TAG=paper_t23_100k GPU_ID=0 SHARD_ID=0 NUM_SHARDS=4 \
#     bash scripts/retrain_table23_100k.sh
#
# Table 2 (GeoGaze):  null / ray / alignment / both
# Table 3 (GazeFlow): direct / parallel / object_hand / hand_object
# "both" and "hand_object" are the same configuration and are trained once.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_TAG="${RUN_TAG:-paper_t23_100k}"
GPU_ID="${GPU_ID:?Set GPU_ID}"
SHARD_ID="${SHARD_ID:?Set SHARD_ID}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SEEDS="${SEEDS:-0 1 2}"
ITERATION="${ITERATION:-100000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LOG_ROOT="outputs/gaze2hoi/retrain_logs/${RUN_TAG}/worker_${SHARD_ID}"
mkdir -p "${LOG_ROOT}"

# setting -> extra hydra overrides ("-" means "defaults only")
settings=(both null ray alignment direct parallel object_hand)
overrides_for() {
  case "$1" in
    both)        echo "" ;;
    null)        echo "gaze2hoi.model.null_gaze_condition=true" ;;
    ray)         echo "gaze2hoi.model.gaze_condition_mode=gage_closeness_temporal" ;;
    alignment)   echo "gaze2hoi.model.gaze_condition_mode=gage_alignment_temporal" ;;
    direct)      echo "gaze2hoi.model.gaze_token_fusion=token" ;;
    parallel)    echo "gaze2hoi.model.cross_attn_order=parallel" ;;
    object_hand) echo "gaze2hoi.model.cross_attn_order=object_hand" ;;
    *) echo "unknown setting $1" >&2; exit 1 ;;
  esac
}

idx=0
for setting in "${settings[@]}"; do
  for seed in ${SEEDS}; do
    if (( idx % NUM_SHARDS != SHARD_ID )); then idx=$((idx+1)); continue; fi
    idx=$((idx+1))
    exp="${RUN_TAG}_${setting}_s${seed}"
    log="${LOG_ROOT}/${exp}.log"
    if [[ -f "outputs/gaze2hoi/${exp}/iteration_0100000.pth" ]]; then
      echo "[skip] ${exp} already has iteration_0100000.pth"; continue
    fi
    echo "[run ] ${exp} (gpu ${GPU_ID})"
    # shellcheck disable=SC2046
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" gaze2hoi/train.py \
      gaze2hoi.exp.name="${exp}" \
      gaze2hoi.exp.iteration="${ITERATION}" \
      gaze2hoi.exp.seed="${seed}" \
      reset=false \
      $(overrides_for "${setting}") \
      > "${log}" 2>&1
    echo "[done] ${exp}"
  done
done
echo "worker ${SHARD_ID} finished"
