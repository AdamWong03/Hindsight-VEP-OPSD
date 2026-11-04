#!/usr/bin/env bash
# CDLSD-VEP v1: single-stage K=4/K=8 LoRA training on the two RTX PRO 6000s.
#
#   bash scripts/train_cdlsd.sh smoke
#   bash scripts/train_cdlsd.sh formal
#   RESUME=/path/to/checkpoint bash scripts/train_cdlsd.sh resume
set -euo pipefail

MODE="${1:-formal}"
case "${MODE}" in smoke|formal|resume) ;; *) echo "usage: $0 [smoke|formal|resume]" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/data1/wyd/envs/mlip/bin/python}"
PACKAGE_DIR="${PACKAGE_DIR:-/data1/wyd/envs/eipd-cu130-packages}"
MODEL="${MODEL:-/data1/wyd/models/Qwen3-VL-8B-Instruct}"
# The local 8B weight directory lacks only processor metadata.  Qwen3-VL 4B/8B
# share the exact tokenizer vocabulary and native image processor, so use the
# complete official 4B processor files while keeping every model weight from 8B.
PROCESSOR="${PROCESSOR:-/data1/wyd/models/Qwen3-VL-4B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/data1/wyd/datasets/EIPD-Data-v1}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/train.jsonl}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT="${OUTPUT:-/data2/wyd/experiments/cdlsd_vep_v1_${MODE}_${STAMP}}"
LOG="${LOG:-${OUTPUT}/train.log}"

[[ -x "${PYTHON}" ]] || { echo "missing python: ${PYTHON}" >&2; exit 2; }
[[ -s "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 2; }
[[ -s "${PROCESSOR}/preprocessor_config.json" ]] || {
  echo "missing processor: ${PROCESSOR}" >&2; exit 2;
}
[[ -s "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 2; }
mkdir -p "${OUTPUT}"

# Pin the two PRO 6000 cards by UUID: this host's CUDA ordinal order has differed
# from nvidia-smi ordering in earlier jobs.
export CUDA_VISIBLE_DEVICES="${CDLSD_CUDA_VISIBLE_DEVICES:-GPU-f48b2c84-f983-64e3-000b-d669b9e64454,GPU-9e62da9a-1b28-25c8-e1a0-e96004a11a95}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PACKAGE_DIR}:${ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DS_SKIP_CUDA_CHECK=1
unset NCCL_P2P_DISABLE NCCL_IB_DISABLE NCCL_SHM_DISABLE NCCL_NET NCCL_PROTO NCCL_ALGO

MAX_STEPS=-1
GRAD_ACCUM=32
EPOCHS=2
SAVE_STEPS=200
ROLLOUT_LOG_EVERY=10
if [[ "${MODE}" == "smoke" ]]; then
  MAX_STEPS="${SMOKE_STEPS:-1}"
  GRAD_ACCUM=1
  EPOCHS=1
  SAVE_STEPS=1
  ROLLOUT_LOG_EVERY=1
fi

args=(
  --model "${MODEL}"
  --processor "${PROCESSOR}"
  --data-root "${DATA_ROOT}"
  --manifest "${MANIFEST}"
  --output "${OUTPUT}"
  --short-steps 4
  --long-steps 8
  --lambda-kd "${LAMBDA_KD:-0.2}"
  --temperature "${TEMPERATURE:-0.8}"
  --top-p "${TOP_P:-0.95}"
  --max-new-tokens "${MAX_NEW_TOKENS:-48}"
  --lr "${LR:-2e-5}"
  --projector-lr "${PROJECTOR_LR:-1e-4}"
  --epochs "${EPOCHS}"
  --max-steps "${MAX_STEPS}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --lora-r 64
  --lora-alpha 128
  --deepspeed "${ROOT}/scripts/zero2.json"
  --gradient-checkpointing
  --checkpoint-prefill
  --image-max-pixels "${IMAGE_MAX_PIXELS:-2097152}"
  --dataloader-num-workers 2
  --logging-steps 1
  --save-steps "${SAVE_STEPS}"
  --save-total-limit 2
  --rollout-log-every "${ROLLOUT_LOG_EVERY}"
  --seed 42
)
if [[ "${MODE}" == "resume" ]]; then
  [[ -n "${RESUME:-}" && -d "${RESUME}" ]] || {
    echo "resume mode requires RESUME=/path/to/checkpoint" >&2; exit 2;
  }
  args+=(--resume-from-checkpoint "${RESUME}")
fi

cat <<EOF
[cdlsd-launch] mode=${MODE}
[cdlsd-launch] GPUs=${CUDA_VISIBLE_DEVICES} global_batch=$((2 * GRAD_ACCUM))
[cdlsd-launch] model=${MODEL}
[cdlsd-launch] manifest=${MANIFEST}
[cdlsd-launch] output=${OUTPUT}
[cdlsd-launch] log=${LOG}
EOF

exec "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=2 --master_port="${MASTER_PORT:-29641}" \
  -m vep_opsd.cdlsd_train "${args[@]}" 2>&1 | tee "${LOG}"
