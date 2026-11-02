#!/usr/bin/env bash
# EIPD v2.0 Stage 1 — pure continuous-latent future prediction (full FT default).
#
#   bash scripts/train_eipd_stage1.sh smoke          # 1 step, LoRA, tiny subset
#   bash scripts/train_eipd_stage1.sh train          # 8 GPUs, full finetune, 3 epochs
#
# Requires cached future features:
#   python scripts/precompute_future_features.py --eipd-root $EIPD_ROOT \
#     --model $MODEL --split train
#
# Options (defaults in brackets; every option also reads the same-named env var):
#   --model DIR              original Qwen3-VL checkpoint
#                            [$MODEL, then models/Qwen3-VL-8B-Instruct]
#   --eipd-root DIR          [$EIPD_ROOT, then $DATA_ROOT/EIPD]
#   --features DIR           [$EIPD_ROOT/future_features/train]
#   --training-mode lora|full    [full for train, lora for smoke]
#   --latent-steps N         K [16]
#   --lambda-v F / --eta F   [0.2 / 0.5]
#   --nproc-per-node N       torchrun ranks [8]
#   --global-batch N         [128] -> grad_accum = N / ranks
#   --grad-accum-steps N     override the derivation
#   --lr F / --latent-lr F   [1e-5 / 1e-4]
#   --epochs F               [3]
#   --deepspeed FILE         [scripts/zero2.json]
#   --gradient-checkpointing on|off  [on]
#   --smoke-samples N        subset size [32]
#   --batch-views 1          share one decode chain between the views. DISABLED by
#                            default: batching changes bf16 kernel selection, which
#                            drifts 2.7% per 4 latent steps, and the 16-step
#                            recurrence turns that into NaN gradients.
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
  --help|-h) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -30; exit 0 ;;
  *) echo "unknown mode '${MODE}' (smoke|train|resume)" >&2; exit 2 ;;
esac

vepopsd_parse_args "model eipd-root manifest features training-mode latent-steps lambda-v eta \
nproc-per-node global-batch grad-accum-steps lr latent-lr epochs deepspeed workers cpu-threads \
gradient-checkpointing smoke-samples image-max-pixels save-steps logging-steps resume output-dir \
dry-run run-name nonfinite batch-views max-steps" "$@"

# Absolute defaults, same policy as _env.sh: no discovery, override by env or flag.
EIPD_ROOT="${EIPD_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD}"
MODEL="${MODEL:-/nfs/ofs-llm-ssd/user/shengrenren_i/models/Qwen3-VL-8B-Instruct}"
MANIFEST="${MANIFEST:-${EIPD_ROOT}/train.jsonl}"
FEATURES="${FEATURES:-${EIPD_ROOT}/future_features/train}"
LATENT_STEPS="${LATENT_STEPS:-16}"
LAMBDA_V="${LAMBDA_V:-0.2}"
ETA="${ETA:-0.5}"
NPROC_PER_NODE="$(vepopsd_nproc)"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
LR="${LR:-1e-5}"
LATENT_LR="${LATENT_LR:-1e-4}"
EPOCHS="${EPOCHS:-3}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-on}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-32}"
WORKERS="${WORKERS:-4}"
CPU_THREADS="${CPU_THREADS:-4}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-2097152}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
RESUME="${RESUME:-}"
DRY_RUN="${DRY_RUN:-0}"

case "${MODE}" in
  smoke) TRAINING_MODE="${TRAINING_MODE:-lora}"; MAX_STEPS=1
         GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" ;;
  train) TRAINING_MODE="${TRAINING_MODE:-full}"; MAX_STEPS="${MAX_STEPS:--1}" ;;
  resume) TRAINING_MODE="${TRAINING_MODE:-full}"; MAX_STEPS="${MAX_STEPS:--1}"; RESUME="${RESUME:-true}" ;;
esac

# Full finetuning of an 8B model with two ~2.5k-token multimodal prefills sits right
# on the 80 GiB ceiling, so the optimizer state lives on the CPU (one Adam step per
# 16 micro-batches, ~2% of wall clock). LoRA fits in plain ZeRO-2.
# `--deepspeed` lands in DEEPSPEED (see _args.sh); the rest of the script uses
# DEEPSPEED_CONFIG, so accept either name.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DEEPSPEED:-}}"
if [[ -z "${DEEPSPEED_CONFIG:-}" ]]; then
  if [[ "${TRAINING_MODE:-full}" == "full" ]]; then
    DEEPSPEED_CONFIG="${SCRIPT_DIR}/zero2.json"
  else
    DEEPSPEED_CONFIG="${SCRIPT_DIR}/zero2.json"
  fi
fi

cd "${ROOT}"
PY="${VEP_OPSD_ENV_PREFIX}/bin/python"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -s "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 2; }
[[ -s "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 2; }

if [[ "${MODE}" == "smoke" ]]; then
  SUBSET="${EIPD_ROOT}/train.smoke${SMOKE_SAMPLES}.jsonl"
  if [[ ! -s "${SUBSET}" ]]; then
    "${PY}" - "${SUBSET}" "${SMOKE_SAMPLES}" "${MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
dst, want, src = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
root = src.parent
kept = []
for line in src.read_text(encoding="utf-8").splitlines():
    if len(kept) >= want:
        break
    row = json.loads(line)
    if all((root / p).is_file() for p in row["past_files"] + [row["bridge_file"]]
           + row["future_files"]):
        kept.append(line)
if len(kept) < want:
    raise SystemExit(f"only {len(kept)}/{want} samples have all frames on disk")
dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(f"wrote {len(kept)} samples -> {dst}")
PY
  fi
  MANIFEST="${SUBSET}"
  SUBSET_NAME="${SUBSET##*/}"; SUBSET_NAME="${SUBSET_NAME%.jsonl}"
  FEATURES="${FEATURES_SMOKE:-${EIPD_ROOT}/future_features/custom_${SUBSET_NAME}}"
  echo "smoke features expected at ${FEATURES}"
fi

[[ -d "${FEATURES}" ]] || {
  echo "missing future features: ${FEATURES}" >&2
  echo "run: python scripts/precompute_future_features.py --eipd-root ${EIPD_ROOT} --model ${MODEL} --split train" >&2
  exit 2; }

DENOM=$((1 * NPROC_PER_NODE))
if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then :; else
  (( GLOBAL_BATCH % DENOM == 0 )) || { echo "global batch ${GLOBAL_BATCH} % nproc ${NPROC_PER_NODE}" >&2; exit 2; }
  GRAD_ACCUM_STEPS=$((GLOBAL_BATCH / DENOM))
fi

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-${HOME}/outputs}/eipd/stage1-${MODE}-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}" "${ROOT}/logs"

args=(--model "${MODEL}" --eipd-root "${EIPD_ROOT}" --manifest "${MANIFEST}"
      --features "${FEATURES}" --output "${OUTPUT_DIR}"
      --training-mode "${TRAINING_MODE}" --latent-steps "${LATENT_STEPS}"
      --lambda-v "${LAMBDA_V}" --eta "${ETA}" --epochs "${EPOCHS}"
      --max-steps "${MAX_STEPS}" --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
      --lr "${LR}" --latent-lr "${LATENT_LR}" --image-max-pixels "${IMAGE_MAX_PIXELS}"
      --dataloader-num-workers "${WORKERS}" --cpu-threads "${CPU_THREADS}"
      --save-steps "${SAVE_STEPS}" --logging-steps "${LOGGING_STEPS}"
      --nonfinite "${NONFINITE:-skip}" --batch-views "${BATCH_VIEWS:-0}")
case "${GRADIENT_CHECKPOINTING}" in
  on|true|1) args+=(--gradient-checkpointing) ;;
  off|false|0) args+=(--no-gradient-checkpointing) ;;
  *) echo "--gradient-checkpointing must be on|off" >&2; exit 2 ;;
esac
case "${DEEPSPEED_CONFIG}" in ""|none|off) ;; *) args+=(--deepspeed "${DEEPSPEED_CONFIG}") ;; esac
[[ -n "${RESUME}" ]] && args+=(--resume-from-checkpoint "${RESUME}")

echo "stage1 mode=${MODE} training_mode=${TRAINING_MODE} nproc=${NPROC_PER_NODE} grad_accum=${GRAD_ACCUM_STEPS} global_batch=$((NPROC_PER_NODE * GRAD_ACCUM_STEPS)) K=${LATENT_STEPS}"
echo "manifest=${MANIFEST}"; echo "features=${FEATURES}"; echo "output=${OUTPUT_DIR}"
vepopsd_dist_args "${NPROC_PER_NODE}"
printf '%q ' "${VEP_OPSD_ENV_PREFIX}/bin/torchrun" "${DIST_ARGS[@]}" -m vep_opsd.eipd_stage1 "${args[@]}"; echo
[[ "${DRY_RUN}" == "1" ]] && { echo "(dry run)"; exit 0; }

"${VEP_OPSD_ENV_PREFIX}/bin/torchrun" "${DIST_ARGS[@]}" -m vep_opsd.eipd_stage1 "${args[@]}" 2>&1 | tee "${ROOT}/logs/stage1-${MODE}-$(date +%Y%m%d-%H%M%S).log"
