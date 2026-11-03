#!/usr/bin/env bash
# 多卡分片评测：每张卡一个进程，跑完自动合并。
#
#   bash scripts/eval_eipd_multi.sh --checkpoint DIR --latent-steps 4
#   bash scripts/eval_eipd_multi.sh --checkpoint DIR --latent-steps 16 --ablate zero
#
# 默认用满可见的卡；验证集 5000 条在 8 卡上约 15-20 分钟一组。
# 想少用卡：GPUS=0,1,2,3 bash scripts/eval_eipd_multi.sh ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"

EIPD_ROOT="${EIPD_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD}"
MODEL="${MODEL:-/nfs/ofs-llm-ssd/user/shengrenren_i/models/Qwen3-VL-8B-Instruct}"
PY="${VEP_OPSD_ENV_PREFIX}/bin/python"

# 参数里必须能拿到 --checkpoint，用来定位日志与合并目录
CHECKPOINT=""
prev=""
for arg in "$@"; do
  [[ "${prev}" == "--checkpoint" ]] && CHECKPOINT="${arg}"
  prev="${arg}"
done
[[ -n "${CHECKPOINT}" ]] || { echo "usage: $0 --checkpoint DIR [other eval args]" >&2; exit 2; }

# 每个分片是一个独立进程，各自把一份 8B 权重读进内存（约 17 GB）。
# 容器内存吃紧时用 PARALLEL 限制同时在跑的进程数。
PARALLEL="${PARALLEL:-0}"        # 0 = 全部并行

if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -r -a DEVICES <<< "${GPUS}"
else
  mapfile -t DEVICES < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
COUNT="${#DEVICES[@]}"
mkdir -p "${ROOT}/logs"

echo "=== 分片评测 ==="
echo "checkpoint : ${CHECKPOINT}"
echo "devices    : ${DEVICES[*]} (${COUNT} 个分片)"
echo "args       : $*"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

declare -a shard_pids=() shard_idx=() shard_log=()
for index in "${!DEVICES[@]}"; do
  device="${DEVICES[index]}"
  log="${ROOT}/logs/eval-$(basename "${CHECKPOINT}")-shard${index}.log"
  if (( PARALLEL > 0 )); then
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do wait -n || true; done
  fi
  CUDA_VISIBLE_DEVICES="${device}" "${PY}" -m vep_opsd.eipd_eval \
    --eipd-root "${EIPD_ROOT}" --base-model "${MODEL}" \
    --shard-index "${index}" --shard-count "${COUNT}" "$@" > "${log}" 2>&1 &
  shard_pids+=($!)
  shard_idx+=("${index}")
  shard_log+=("${log}")
  echo "  gpu ${device} -> 分片 ${index}/${COUNT}  pid ${shard_pids[-1]}  log ${log}"
done

declare -a failed_shards=()
for position in "${!shard_pids[@]}"; do
  if ! wait "${shard_pids[position]}"; then
    failed_shards+=("${position}")
  fi
done
if (( ${#failed_shards[@]} > 0 )); then
  echo "=== ${#failed_shards[@]}/${COUNT} 个分片失败 ===" >&2
  for position in "${failed_shards[@]}"; do
    echo "--- 分片 ${shard_idx[position]} (gpu ${DEVICES[position]}) 尾部:" >&2
    tail -n 12 "${shard_log[position]}" >&2 || true
  done
  exit 1
fi

# 合并：从参数里取出 ablate / latent-steps 组成与分片一致的 tag
ABLATE="none"; K="4"; prev=""
for arg in "$@"; do
  [[ "${prev}" == "--ablate" ]] && ABLATE="${arg}"
  [[ "${prev}" == "--latent-steps" ]] && K="${arg}"
  prev="${arg}"
done
"${PY}" -m vep_opsd.eipd_eval --merge "${CHECKPOINT}" --checkpoint "${CHECKPOINT}" \
    --tag "${ABLATE}-k${K}"
