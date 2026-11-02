#!/usr/bin/env bash
# EIPD v2.0 Stage 2 — frozen dual-condition teacher, LoRA-only student.
#
#   bash scripts/train_eipd_stage2.sh smoke --stage1 /runs/stage1 --base-model $MODEL
#   bash scripts/train_eipd_stage2.sh train --stage1 /runs/stage1 --base-model $MODEL
#
# Stage 2 keeps the Stage-1 weights, the latent projector and the value
# projection frozen; only the Student LoRA adapter trains. If `--stage1` holds a
# full checkpoint (config.json) the base model is taken from there, otherwise
# pass --base-model.
#
# Options (defaults in brackets; every option also reads the same-named env var):
#   --stage1 DIR             Stage 1 output directory [required]
#   --base-model DIR         original Qwen3-VL dir when Stage 1 saved an adapter only
#   --eipd-root DIR          [$EIPD_ROOT]
#   --features DIR           [$EIPD_ROOT/future_features/train]
#   --latent-steps N         K [16]
#   --lambda-v F / --lambda-d F / --eta F / --alpha F   [0.2 / 0.2 / 0.5 / 0.5]
#   --nproc-per-node N       [8]
#   --global-batch N         [128] -> grad_accum = N / ranks
#   --lr F                   [2e-5]
#   --epochs F               [2]
#   --deepspeed FILE         [scripts/zero2.json]
#   --teacher-check-every N  teacher logits invariant every N micro-steps [50]
#   --kd-mode gain|js        gain = only distil samples whose Bridge improved the gold
#                            likelihood (paper option); js = weight by disagreement
#   --dry-run 1              print the command, run nothing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_args.sh"
source "${SCRIPT_DIR}/_env.sh"
# shellcheck source=scripts/_dist.sh
source "${SCRIPT_DIR}/_dist.sh"

MODE="${1:-train}"
case "${MODE}" in
  smoke|train|resume) shift || true ;;
  --help|-h) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -28; exit 0 ;;
  *) echo "unknown mode '${MODE}' (smoke|train|resume)" >&2; exit 2 ;;
esac

vepopsd_parse_args "stage1 base-model eipd-root manifest features latent-steps lambda-v \
lambda-d eta alpha nproc-per-node global-batch grad-accum-steps lr epochs deepspeed workers \
cpu-threads gradient-checkpointing smoke-samples image-max-pixels save-steps logging-steps \
teacher-check-every resume output-dir dry-run nonfinite kd-mode" "$@"

EIPD_ROOT="${EIPD_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD}"
MANIFEST="${MANIFEST:-${EIPD_ROOT}/train.jsonl}"
FEATURES="${FEATURES:-${EIPD_ROOT}/future_features/train}"
LATENT_STEPS="${LATENT_STEPS:-16}"
LAMBDA_V="${LAMBDA_V:-0.2}"
LAMBDA_D="${LAMBDA_D:-0.2}"
ETA="${ETA:-0.5}"
ALPHA="${ALPHA:-0.5}"
NPROC_PER_NODE="$(vepopsd_nproc)"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-on}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-32}"
WORKERS="${WORKERS:-4}"
CPU_THREADS="${CPU_THREADS:-4}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-${SCRIPT_DIR}/zero2.json}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-2097152}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
TEACHER_CHECK_EVERY="${TEACHER_CHECK_EVERY:-50}"
RESUME="${RESUME:-}"
DRY_RUN="${DRY_RUN:-0}"

case "${MODE}" in
  smoke) MAX_STEPS=1; GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" ;;
  train) MAX_STEPS="${MAX_STEPS:--1}" ;;
  resume) MAX_STEPS="${MAX_STEPS:--1}"; RESUME="${RESUME:-true}" ;;
esac

[[ -n "${STAGE1:-}" && -d "${STAGE1}" ]] || { echo "--stage1 DIR is required" >&2; exit 2; }
cd "${ROOT}"
PY="${VEP_OPSD_ENV_PREFIX}/bin/python"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -s "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 2; }

if [[ "${MODE}" == "smoke" ]]; then
  SUBSET="${EIPD_ROOT}/train.smoke${SMOKE_SAMPLES}.jsonl"
  [[ -s "${SUBSET}" ]] || { echo "run stage1 smoke first to build ${SUBSET}" >&2; exit 2; }
  MANIFEST="${SUBSET}"
  SUBSET_NAME="${SUBSET##*/}"; SUBSET_NAME="${SUBSET_NAME%.jsonl}"
  FEATURES="${FEATURES_SMOKE:-${EIPD_ROOT}/future_features/custom_${SUBSET_NAME}}"
fi

[[ -d "${FEATURES}" ]] || { echo "missing future features: ${FEATURES}" >&2; exit 2; }

DENOM=$((1 * NPROC_PER_NODE))
if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then :; else
  (( GLOBAL_BATCH % DENOM == 0 )) || { echo "global batch ${GLOBAL_BATCH} % nproc ${NPROC_PER_NODE}" >&2; exit 2; }
  GRAD_ACCUM_STEPS=$((GLOBAL_BATCH / DENOM))
fi

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-${HOME}/outputs}/eipd/stage2-${MODE}-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}" "${ROOT}/logs"

args=(--stage1 "${STAGE1}" --eipd-root "${EIPD_ROOT}" --manifest "${MANIFEST}"
      --features "${FEATURES}" --output "${OUTPUT_DIR}" --latent-steps "${LATENT_STEPS}"
      --lambda-v "${LAMBDA_V}" --lambda-d "${LAMBDA_D}" --eta "${ETA}" --alpha "${ALPHA}"
      --epochs "${EPOCHS}" --max-steps "${MAX_STEPS}"
      --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" --lr "${LR}"
      --image-max-pixels "${IMAGE_MAX_PIXELS}" --dataloader-num-workers "${WORKERS}"
      --cpu-threads "${CPU_THREADS}" --save-steps "${SAVE_STEPS}"
      --logging-steps "${LOGGING_STEPS}" --teacher-check-every "${TEACHER_CHECK_EVERY}"
      --kd-mode "${KD_MODE:-js}")
[[ -n "${BASE_MODEL:-}" ]] && args+=(--base-model "${BASE_MODEL}")
case "${GRADIENT_CHECKPOINTING}" in
  on|true|1) args+=(--gradient-checkpointing) ;;
  off|false|0) args+=(--no-gradient-checkpointing) ;;
  *) echo "--gradient-checkpointing must be on|off" >&2; exit 2 ;;
esac
case "${DEEPSPEED_CONFIG}" in ""|none|off) ;; *) args+=(--deepspeed "${DEEPSPEED_CONFIG}") ;; esac
[[ -n "${RESUME}" ]] && args+=(--resume-from-checkpoint "${RESUME}")

echo "stage2 mode=${MODE} nproc=${NPROC_PER_NODE} grad_accum=${GRAD_ACCUM_STEPS} global_batch=$((NPROC_PER_NODE * GRAD_ACCUM_STEPS)) K=${LATENT_STEPS}"
echo "stage1=${STAGE1}"; echo "manifest=${MANIFEST}"; echo "output=${OUTPUT_DIR}"
vepopsd_dist_args "${NPROC_PER_NODE}"
printf '%q ' "${VEP_OPSD_ENV_PREFIX}/bin/torchrun" "${DIST_ARGS[@]}" -m vep_opsd.eipd_stage2 "${args[@]}"; echo
[[ "${DRY_RUN}" == "1" ]] && { echo "(dry run)"; exit 0; }

"${VEP_OPSD_ENV_PREFIX}/bin/torchrun" "${DIST_ARGS[@]}" -m vep_opsd.eipd_stage2 "${args[@]}" 2>&1 | tee "${ROOT}/logs/stage2-${MODE}-$(date +%Y%m%d-%H%M%S).log"
