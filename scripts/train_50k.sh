#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL:?Set MODEL to the Qwen3-VL-8B-Instruct directory}"
: "${DATA:?Set DATA to the compact 50K JSON manifest}"
: "${VIDEO_ROOT:?Set VIDEO_ROOT so relative manifest video paths resolve}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to durable storage}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
DENOM=$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % DENOM != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by NPROC_PER_NODE*PER_DEVICE_BATCH_SIZE" >&2
  exit 2
fi
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / DENOM))

cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT:-29651}" \
  -m vep_opsd.train \
  --model_name_or_path "${MODEL}" \
  --manifest "${DATA}" \
  --video_root "${VIDEO_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed "${DEEPSPEED_CONFIG:-scripts/zero2.json}" \
  --bf16 True \
  --tf32 False \
  --gradient_checkpointing True \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --num_train_epochs 1 \
  --learning_rate "${LR:-1e-5}" \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-50}" \
  --save_total_limit 2 \
  --remove_unused_columns False \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}" \
  --ddp_find_unused_parameters False \
  --report_to none \
  --seed 42 \
  --candidates_per_example 4 \
  --sampling_temperature 0.8 \
  --sampling_top_p 0.95 \
  --teacher_temperature 0.2 \
  --student_temperature 1.0 \
  --minimum_consistency 0.6 \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}"

