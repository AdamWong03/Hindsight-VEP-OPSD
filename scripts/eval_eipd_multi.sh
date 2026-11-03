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

pids=()
for index in "${!DEVICES[@]}"; do
  device="${DEVICES[index]}"
  log="${ROOT}/logs/eval-$(basename "${CHECKPOINT}")-shard${index}.log"
  CUDA_VISIBLE_DEVICES="${device}" "${PY}" -m vep_opsd.eipd_eval \
    --eipd-root "${EIPD_ROOT}" --base-model "${MODEL}" \
    --shard-index "${index}" --shard-count "${COUNT}" "$@" > "${log}" 2>&1 &
  pids+=($!)
  echo "  gpu ${device} -> 分片 ${index}/${COUNT}  pid ${pids[-1]}  log ${log}"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
if (( failed > 0 )); then
  echo "${failed}/${COUNT} 个分片失败，查看 logs/eval-*.log" >&2
  exit 1
fi

"${PY}" -m vep_opsd.eipd_eval --merge "${CHECKPOINT}"
