#!/usr/bin/env bash
# Hindsight VEP-OPSD v1 — 2x A6000 smoke run.
#
# Small end-to-end Generate -> Privileged Teacher -> Token Distill run.
#
#   bash scripts/train_smoke.sh
#   bash scripts/train_smoke.sh --max-steps 4 --max-new-tokens 64
#
# The subset is carved out of $DATA, so MODEL/DATA must come from .env (see
# .env.example) or from the flags.
#
# Options (everything else is forwarded to scripts/train_50k.sh):
#   --smoke-dir DIR       where the subset lives [$OUTPUT_ROOT/.../smoke-data]
#   --smoke-samples N     subset size [200]
#   --max-steps N         optimizer steps to run [8]
#   --max-new-tokens N    rollout length [128]
#   --output-dir DIR      [$OUTPUT_ROOT/hindsight-vep-opsd/smoke-<ts>]
#   --deepspeed-config F  [scripts/zero2_offload.json]
#   --nproc-per-node N    [2]
#   --env-name/--env-prefix/--config  as in train_50k.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/_args.sh
source "${SCRIPT_DIR}/_args.sh"

vepopsd_usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -20; }

vepopsd_parse_args \
  "smoke-dir smoke-samples max-steps max-new-tokens output-dir deepspeed-config \
nproc-per-node env-name env-prefix config" "$@"

# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "${ROOT}"

# The offload path needs DeepSpeed's CPU Adam, which is JIT compiled. The local
# node ships nvcc 13.1 against a torch built with 13.0, and DeepSpeed refuses to
# build on that mismatch; the minor difference is harmless for cpu_adam.
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

SMOKE_DIR="${SMOKE_DIR:-${OUTPUT_ROOT:-${HOME}/outputs}/hindsight-vep-opsd/smoke-data}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-200}"
mkdir -p "${SMOKE_DIR}"

: "${DATA:?Set --data or DATA to the training manifest (see .env.example)}"
SOURCE_DATA="${DATA}"
SUBSET="${SMOKE_DIR}/vep.smoke${SMOKE_SAMPLES}.json"
if [[ ! -s "${SUBSET}" ]]; then
  echo "building ${SUBSET} from ${SOURCE_DATA}"
  python - "${SUBSET}" "${SMOKE_SAMPLES}" "${SOURCE_DATA}" <<'PY'
import json
import sys
from pathlib import Path

dst = Path(sys.argv[1])
want = int(sys.argv[2])
src = Path(sys.argv[3])
rows = json.loads(src.read_text())
present = []
for row in rows:
    if len(present) >= want:
        break
    if Path(str(row.get('video', ''))).is_file():
        present.append(row)
if len(present) < 8:
    raise SystemExit(f'only {len(present)} samples have a local video')
dst.write_text(json.dumps(present, ensure_ascii=False))
print(f'wrote {len(present)} samples -> {dst}')
PY
fi

exec bash "${SCRIPT_DIR}/train_50k.sh" \
  --data "${SUBSET}" \
  --nproc-per-node "${NPROC_PER_NODE:-2}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE:-2}" \
  --output-dir "${OUTPUT_DIR:-${OUTPUT_ROOT:-${HOME}/outputs}/hindsight-vep-opsd/smoke-$(date +%Y%m%d-%H%M%S)}" \
  --max-steps "${MAX_STEPS:-8}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --save-steps "${SAVE_STEPS:-4}" \
  --save-total-limit 1 \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-1}" \
  --deepspeed-config "${DEEPSPEED_CONFIG:-scripts/zero2_offload.json}" \
  --run-name "${RUN_NAME:-smoke-vep-opsd}" \
  --env-prefix "${VEP_OPSD_ENV_PREFIX}"
