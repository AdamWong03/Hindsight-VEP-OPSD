#!/usr/bin/env bash
# Precompute the frozen future-frame features with one worker per GPU.
#
#   bash scripts/precompute_future_features_multi.sh                 # all visible GPUs
#   bash scripts/precompute_future_features_multi.sh --gpus 0,1,2,3  # a subset
#
# Each worker owns one GPU, handles a contiguous block of shard indices and skips
# shards that already exist, so the job is restartable and re-runnable.
#
# Options (defaults in brackets):
#   --gpus LIST          CUDA devices, comma separated [all visible]
#   --eipd-root DIR      [$EIPD_ROOT, then $DATA_ROOT/EIPD]
#   --model DIR          original Qwen3-VL checkpoint [models/Qwen3-VL-8B-Instruct]
#   --split NAME         train | val [train]
#   --shard-size N       samples per shard [1000]
#   --cpu-threads N      torch threads per worker [4]
#   --vision-dump FILE   cached vision tower [_vision_tower.pt next to the features]
#   --dry-run 1          print the worker plan, run nothing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/_args.sh
source "${SCRIPT_DIR}/_args.sh"
# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"

vepopsd_parse_args "gpus eipd-root model split shard-size cpu-threads vision-dump dry-run" "$@"

# Absolute defaults, same policy as _env.sh: no discovery, override by env or flag.
EIPD_ROOT="${EIPD_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD}"
MODEL="${MODEL:-/nfs/ofs-llm-ssd/user/shengrenren_i/models/Qwen3-VL-8B-Instruct}"
SPLIT="${SPLIT:-train}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
CPU_THREADS="${CPU_THREADS:-4}"
PY="${VEP_OPSD_ENV_PREFIX}/bin/python"
FEATURE_ROOT="${EIPD_ROOT}/future_features"
VISION_DUMP="${VISION_DUMP:-${FEATURE_ROOT}/_vision_tower.pt}"
DRY_RUN="${DRY_RUN:-0}"

cd "${ROOT}"
[[ -s "${EIPD_ROOT}/${SPLIT}.jsonl" ]] || { echo "missing ${EIPD_ROOT}/${SPLIT}.jsonl" >&2; exit 2; }
[[ -s "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 2; }

if [[ -z "${GPUS:-}" ]]; then
  GPUS="$("${PY}" - <<'PY'
import torch
print(",".join(str(i) for i in range(torch.cuda.device_count())))
PY
)"
fi
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
WORKERS="${#GPU_LIST[@]}"
(( WORKERS > 0 )) || { echo "no GPUs selected" >&2; exit 2; }

SAMPLES=$(wc -l < "${EIPD_ROOT}/${SPLIT}.jsonl")
SHARDS=$(( (SAMPLES + SHARD_SIZE - 1) / SHARD_SIZE ))
PER_WORKER=$(( (SHARDS + WORKERS - 1) / WORKERS ))

echo "samples=${SAMPLES} shard_size=${SHARD_SIZE} shards=${SHARDS} workers=${WORKERS} per_worker=${PER_WORKER}"
echo "features -> ${FEATURE_ROOT}/${SPLIT}"
mkdir -p "${FEATURE_ROOT}/${SPLIT}" "${ROOT}/logs"

if [[ ! -f "${VISION_DUMP}" ]]; then
  echo "dumping vision tower once (one full checkpoint load, ~40 s)"
  "${PY}" "${SCRIPT_DIR}/precompute_future_features.py" --eipd-root "${EIPD_ROOT}" \
    --model "${MODEL}" --dump-vision-to "${VISION_DUMP}"
fi

export OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}"
pids=()
names=()
for ((i = 0; i < WORKERS; i++)); do
  start=$(( i * PER_WORKER ))
  end=$(( start + PER_WORKER ))
  (( end > SHARDS )) && end=${SHARDS}
  if (( start >= SHARDS )); then continue; fi
  gpu="${GPU_LIST[$i]}"
  log="${ROOT}/logs/features-${SPLIT}-gpu${gpu}.log"
  echo "worker ${i}: gpu ${gpu} shards [${start}, ${end}) -> ${log}"
  if [[ "${DRY_RUN}" == "1" ]]; then continue; fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" -u "${SCRIPT_DIR}/precompute_future_features.py" \
    --eipd-root "${EIPD_ROOT}" --model "${MODEL}" --split "${SPLIT}" \
    --shard-size "${SHARD_SIZE}" --shard-start "${start}" --shard-end "${end}" \
    --vision-dump "${VISION_DUMP}" --cpu-threads "${CPU_THREADS}" --device cuda:0 \
    > "${log}" 2>&1 &
  pids+=($!)
  names+=("gpu${gpu}")
done

[[ "${DRY_RUN}" == "1" ]] && { echo "(dry run)"; exit 0; }

status=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    echo "worker ${names[$idx]}: ok"
  else
    echo "worker ${names[$idx]}: FAILED (see logs/features-${SPLIT}-${names[$idx]}.log)" >&2
    status=1
  fi
done

have=$(ls "${FEATURE_ROOT}/${SPLIT}"/shard_*.pt 2>/dev/null | wc -l)
echo "shards present: ${have}/${SHARDS}"
[[ "${have}" -lt "${SHARDS}" ]] && echo "some shards are missing; re-run to fill them" >&2
exit "${status}"
