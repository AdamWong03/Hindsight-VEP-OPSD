#!/usr/bin/env bash
# Formal EIPD full-LM finetuning on the two RTX PRO 6000 cards (physical 0,3).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/data1/wyd/envs/cfsr/bin/python}"
MODEL="${MODEL:-/data1/hf-models/Qwen3-VL-8B-Instruct}"
MANIFEST="${MANIFEST:-/data1/wyd/datasets/EIPD-Data-v1/train.jsonl}"
OUTPUT="${OUTPUT:-/data1/wyd/experiments/eipd_qwen3vl8b_full50k_fullft_s42_v1_2xpro}"
LOG="${LOG:-${OUTPUT}/train.log}"

mkdir -p "${OUTPUT}"
# Pin the physical PRO 6000 cards by UUID. This host's default CUDA ordinal
# order differs from nvidia-smi's index order, so numeric 0,3 is not reliable.
# Override explicitly with EIPD_CUDA_VISIBLE_DEVICES when another pair is wanted.
export CUDA_VISIBLE_DEVICES="${EIPD_CUDA_VISIBLE_DEVICES:-GPU-f48b2c84-f983-64e3-000b-d669b9e64454,GPU-9e62da9a-1b28-25c8-e1a0-e96004a11a95}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON}" -m torch.distributed.run --nproc_per_node=1 --master_port="${MASTER_PORT:-29631}" \
  -m vep_opsd.eipd_train \
  --model "${MODEL}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT}" \
  --training-mode full \
  --teacher-device cuda:1 \
  --deepspeed scripts/zero3_optimizer_offload.json \
  --epochs 1 \
  --lr 1e-5 \
  --weight-decay 0.1 \
  --warmup-ratio 0.1 \
  --lr-scheduler-type cosine \
  --gradient-accumulation-steps 128 \
  --max-new-tokens 256 \
  --max-answer-tokens 128 \
  --image-max-pixels 2097152 \
  --lambda-d 0.2 \
  --alpha 0.5 \
  --seed 42 \
  --gradient-checkpointing \
  --logging-steps 1 \
  --save-strategy steps \
  --save-steps 200 \
  --save-total-limit 1 \
  --save-only-model \
  2>&1 | tee -a "${LOG}"
