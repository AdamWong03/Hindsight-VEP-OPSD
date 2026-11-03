#!/usr/bin/env bash
# Latent-only VEP evaluation: (Past, Q) -> Z -> answer, no Bridge / future / gold prefix.
#
#   bash scripts/eval_eipd.sh --checkpoint <stage1 或 stage2 输出目录>
#   bash scripts/eval_eipd.sh --checkpoint DIR --ablate zero      # Z := 0
#   bash scripts/eval_eipd.sh --checkpoint DIR --ablate shuffle   # 换成别的样本的 Z
#   bash scripts/eval_eipd.sh --checkpoint DIR --limit 200        # 先跑一小批
#
# 三个数字要一起看：正常 / zero / shuffle。若后两者几乎不掉，说明答案没有真正经过 latent。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"

EIPD_ROOT="${EIPD_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD}"
MODEL="${MODEL:-/nfs/ofs-llm-ssd/user/shengrenren_i/models/Qwen3-VL-8B-Instruct}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
exec "${VEP_OPSD_ENV_PREFIX}/bin/python" -m vep_opsd.eipd_eval \
  --eipd-root "${EIPD_ROOT}" --base-model "${MODEL}" "$@"
