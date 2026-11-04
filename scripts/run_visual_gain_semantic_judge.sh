#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?pilot or full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=/data2/wyd/experiments/vep_visual_gain_audit_ckpt400_20261105
MODEL=/data2/wyd/models/Qwen2.5-32B-Instruct
OUTPUT="${SEMANTIC_REVIEW_OUTPUT:-$RUN/semantic-judge}"
RUBRIC="${SEMANTIC_RUBRIC:-initial}"
export PYTHONPATH="/data1/wyd/envs/eipd-cu130-packages:$ROOT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=GPU-f48b2c84-f983-64e3-000b-d669b9e64454
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/data2/wyd/tmp
while systemctl --user is-active --quiet visual-gain-judge-stage; do
  echo "[$(date -Is)] waiting for local 32B weight staging"
  sleep 30
done
if systemctl --user is-failed --quiet visual-gain-judge-stage; then
  echo 'Local judge staging failed; refusing incomplete weights' >&2
  exit 2
fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
source=Path('/data1/wyd/models/Qwen2.5-32B-Instruct')
target=Path('/data2/wyd/models/Qwen2.5-32B-Instruct')
index=json.loads((target/'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    if not (target/name).is_file() or (target/name).stat().st_size != (source/name).stat().st_size:
        raise RuntimeError(f'incomplete local model shard: {name}')
print('All local independent-judge shards present with matching original sizes',flush=True)
PY
while true; do
  free="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)"
  if (( free >= 81920 )); then break; fi
  echo "[$(date -Is)] waiting for 80 GiB free on PRO 0; currently ${free}MiB"
  sleep 30
done
if [[ "$MODE" == full && ! -f "$OUTPUT/pilot-approved.json" ]]; then
  echo 'Full semantic review requires recorded pilot inspection; refusing unattended promotion' >&2
  exit 2
fi
cd "$ROOT"
exec /data1/wyd/envs/mlip/bin/python -u -m vep_opsd.visual_gain_judge \
  --run "$RUN" --model "$MODEL" --output "$OUTPUT" \
  --mode "$MODE" --batch-size 16 --max-new-tokens 128 --rubric "$RUBRIC"
