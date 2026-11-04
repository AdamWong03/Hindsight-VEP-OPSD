#!/usr/bin/env bash
# Persistent worker, invoked by systemd-run --user. All artifacts live on data2.
set -euo pipefail
FAMILY="${1:?base or cdlsd}"
SHARD="${2:?shard index}"
COUNT="${3:?shard count}"
GPU="${4:?GPU UUID}"
MODE="${5:-full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=/data2/wyd/experiments/vep_visual_gain_audit_ckpt400_20261105
DATA=/data2/wyd/datasets/EIPD-Data-v1-visual-gain-val5k
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="/data1/wyd/envs/eipd-cu130-packages:$ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/data2/wyd/tmp
mkdir -p "$RUN/$MODE" "$TMPDIR"
printf '[%s] family=%s shard=%s/%s GPU=%s mode=%s\n' "$(date -Is)" "$FAMILY" "$SHARD" "$COUNT" "$GPU" "$MODE"
while [[ ! -s "$DATA/data-config.json" ]]; do
  [[ "$(systemctl --user show vep-visual-gain-data -p ActiveState --value)" != failed ]] || exit 2
  echo "[$(date -Is)] waiting for complete staged data"
  sleep 15
done
if [[ "$MODE" == full ]]; then
  while true; do
    free="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)"
    if (( free >= 26624 )); then break; fi
    echo "[$(date -Is)] waiting for GPU=$GPU free=${free}MiB minimum=26624MiB"
    sleep 30
  done
fi
args=()
if [[ "$MODE" == pilot* ]]; then
  args+=(--limit-per-category 3 --verify-caches)
fi
/data1/wyd/envs/mlip/bin/python -u -m vep_opsd.visual_gain_eval \
  --family "$FAMILY" --base-model /data2/wyd/models/Qwen3-VL-8B-Instruct \
  --processor /data1/wyd/models/Qwen3-VL-4B-Instruct \
  --checkpoint /data2/wyd/experiments/cdlsd_vep_v1_formal_50k_s42_20261104-010215/checkpoint-400 \
  --data-root "$DATA" --output "$RUN/$MODE" \
  --shard-index "$SHARD" --shard-count "$COUNT" "${args[@]}"
touch "$RUN/$MODE/$FAMILY-$SHARD.complete"
