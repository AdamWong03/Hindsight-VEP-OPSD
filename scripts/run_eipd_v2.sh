#!/usr/bin/env bash
# EIPD v2.0 — one-shot pipeline. Temporary convenience wrapper; every step is
# also runnable on its own (see docs/eipd-runbook.md).
#
#   bash scripts/run_eipd_v2.sh smoke     # features(32) + stage1 smoke + stage2 smoke   ~15 min
#   bash scripts/run_eipd_v2.sh full      # features(50k) + stage1 full + stage2 full     hours
#
# Environment overrides:
#   MODEL        original Qwen3-VL dir      [<data>/../models/Qwen3-VL-8B-Instruct]
#   EIPD_ROOT    dataset root               [$DATA_ROOT/EIPD]
#   OUT_BASE     where runs go              [<OUTPUT_ROOT>/eipd/v2-<mode>-<ts>]
#   NPROC_PER_NODE  cards                   [auto from torch.cuda.device_count()]
#   TRAINING_MODE lora|full                 [full for `full`, lora for `smoke`]
#   GLOBAL_BATCH  optimizer batch           [128]
#   SKIP_FEATURES=1 reuse an existing cache [off]
#
# Every step is restartable: existing shards and checkpoint dirs are reused.
set -euo pipefail

MODE="${1:-smoke}"
case "${MODE}" in
  smoke|full) ;;
  *) echo "usage: $0 smoke|full" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/_env.sh
# <repo>/.env holds the machine paths; without it DATA_ROOT/MODEL fall back to
# nonsense like "/EIPD". Loaded before anything derives a default from it.
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"
# shellcheck source=scripts/_dist.sh
source "${SCRIPT_DIR}/_dist.sh"

# Absolute defaults, same policy as _env.sh: no discovery, override by env or flag.
EIPD_ROOT="${EIPD_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD}"
MODEL="${MODEL:-/nfs/ofs-llm-ssd/user/shengrenren_i/models/Qwen3-VL-8B-Instruct}"
PY="${VEP_OPSD_ENV_PREFIX}/bin/python"
NPROC="$(vepopsd_nproc)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_BASE="${OUT_BASE:-${OUTPUT_ROOT:-${HOME}/outputs}/eipd/v2-${MODE}-${STAMP}}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-32}"
mkdir -p "${OUT_BASE}" "${ROOT}/logs"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

say() { echo; echo "===== $* ====="; }

say "plan"
echo "mode=${MODE} nproc=${NPROC} model=${MODEL}"
echo "eipd_root=${EIPD_ROOT}"
echo "out=${OUT_BASE}"
[[ -s "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 2; }
[[ -s "${EIPD_ROOT}/train.jsonl" ]] || { echo "missing manifest under ${EIPD_ROOT}" >&2; exit 2; }
"${PY}" -c "import torch;print('gpus:', torch.cuda.device_count(), '| free GiB:',
  [round((torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i))/2**30,1)
   for i in range(torch.cuda.device_count())])" 2>/dev/null || true

if [[ "${MODE}" == "smoke" ]]; then
  MANIFEST="${EIPD_ROOT}/train.smoke${SMOKE_SAMPLES}.jsonl"
  FEATURES="${EIPD_ROOT}/future_features/custom_train.smoke${SMOKE_SAMPLES}"
  TRAINING_MODE="${TRAINING_MODE:-lora}"
  MAX_STEPS=1
else
  MANIFEST="${EIPD_ROOT}/train.jsonl"
  FEATURES="${EIPD_ROOT}/future_features/train"
  TRAINING_MODE="${TRAINING_MODE:-full}"
  MAX_STEPS=-1
fi

say "1/4 future features -> ${FEATURES}"
if [[ "${SKIP_FEATURES:-0}" == "1" ]]; then
  echo "skipped (SKIP_FEATURES=1)"
elif [[ -d "${FEATURES}" ]] && compgen -G "${FEATURES}/shard_*.pt" >/dev/null; then
  echo "reusing cached shards"
else
  feature_args=(--eipd-root "${EIPD_ROOT}" --model "${MODEL}" --split train
                --shard-size 2000)
  [[ "${MODE}" == "smoke" ]] && feature_args+=(--manifest "${MANIFEST}")
  "${PY}" "${ROOT}/scripts/precompute_future_features.py" "${feature_args[@]}"
fi
[[ -d "${FEATURES}" ]] || { echo "feature cache missing: ${FEATURES}" >&2; exit 2; }

say "2/4 stage 1 (${TRAINING_MODE})"
STAGE1_DIR="${OUT_BASE}/stage1"
EXTRA=()
if (( MAX_STEPS > 0 )); then EXTRA+=(--max-steps "${MAX_STEPS}"); fi
if [[ "${MODE}" == "smoke" ]]; then EXTRA+=(--grad-accum-steps 1 --save-steps 1); else EXTRA+=(--save-steps 500); fi

bash "${SCRIPT_DIR}/train_eipd_stage1.sh" train \
  --model "${MODEL}" --eipd-root "${EIPD_ROOT}" --manifest "${MANIFEST}" \
  --features "${FEATURES}" --output-dir "${STAGE1_DIR}" \
  --training-mode "${TRAINING_MODE}" --nproc-per-node "${NPROC}" "${EXTRA[@]}"

say "3/4 stage 2 (LoRA student, frozen teacher)"
STAGE2_DIR="${OUT_BASE}/stage2"
bash "${SCRIPT_DIR}/train_eipd_stage2.sh" train \
  --stage1 "${STAGE1_DIR}" --base-model "${MODEL}" \
  --eipd-root "${EIPD_ROOT}" --manifest "${MANIFEST}" --features "${FEATURES}" \
  --output-dir "${STAGE2_DIR}" --nproc-per-node "${NPROC}" "${EXTRA[@]}"

say "4/4 done"
cat <<EOF
stage 1 : ${STAGE1_DIR}
stage 2 : ${STAGE2_DIR}
logs    : ${ROOT}/logs/stage{1,2}-*.log

What to look at in the logs:
  stage1: ce_past, ce_bridge, state_loss, temporal_loss, forced_anchors
  stage2: eipd_loss, bridge_js, gold_logprob_gain, teacher_logit_drift (must stay ~0)
          and the share of samples with bridge_js high but gold_logprob_gain < 0
EOF
