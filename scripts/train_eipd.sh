#!/usr/bin/env bash
# EIPD (text-token) training launcher — original Qwen3-VL + EIPD-Data-v1.
#
#   bash scripts/train_eipd.sh smoke --model /models/Qwen3-VL-8B-Instruct    # 1 step
#   bash scripts/train_eipd.sh train --model /models/Qwen3-VL-8B-Instruct    # 2 GPUs, LoRA
#   bash scripts/train_eipd.sh resume --model /models/Qwen3-VL-8B-Instruct   # newest checkpoint
#
# The entry point takes the *original* Qwen3-VL checkpoint. It rejects
# Future-L1 weights (any config with latent_id / max_latent_token): this EIPD
# variant distils text tokens, not continuous latents.
#
# Options (defaults in brackets; every option also reads the same-named env var):
#   --model DIR              original Qwen3-VL checkpoint
#                            [$MODEL, then models/Qwen3-VL-8B-Instruct on this host]
#   --processor DIR          separate processor dir if the weight copy lacks one
#   --training-mode lora|full    [lora] full needs a second frozen copy of 8B
#   --teacher-device cuda:N  only for --training-mode full [unset = same device]
#   --eipd-root DIR          dataset root [$EIPD_ROOT, then $DATA_ROOT/EIPD]
#   --manifest FILE          [$EIPD_ROOT/train.jsonl]
#   --output-dir DIR         [$OUTPUT_ROOT/eipd/<run>-<ts>]
#   --nproc-per-node N       torchrun ranks [2]; even 1 goes through torchrun so
#                            transformers does not pick DataParallel (which doubles
#                            the batch and trips the batch-1 collator)
#   --global-batch N         optimizer batch, Future-L1 convention [128]; with
#                            per-device batch 1 this fixes grad_accum = N/ranks
#   --grad-accum-steps N     override the derivation from --global-batch
#   --max-steps N            [-1 = one epoch]
#   --epochs F               [1]
#   --lr F                   [1e-5]
#   --adam-beta2 F           [0.95, the Future-L1 SFT convention]
#   --max-new-tokens N       rollout length [256]
#   --max-answer-tokens N    [128]
#   --image-max-pixels N     [2097152] (2 MPix; the frames are <=1.28 MPix)
#   --lora-r N / --lora-alpha N   [64 / 128]
#   --lambda-d F / --alpha F [0.2 / 0.5]
#   --deepspeed FILE         ZeRO json [scripts/zero2.json]
#   --workers N              dataloader workers [6]
#   --cpu-threads N          torch CPU threads [4]; 64 is ~5x slower on preprocessing
#   --gradient-checkpointing on|off  [off for lora, on for full]
#   --save-steps N / --logging-steps N   [200 / 10]
#   --resume PATH            explicit checkpoint dir [off]
#   --smoke-samples N        smoke subset size [32]
#   --subset-samples N       train on the first N usable samples (0 = full manifest) [0]
#   --dry-run 1              print the command, run nothing
#
# Batch size is 1 per rank by contract (the EIPD collator rejects >1); raise the
# effective batch with --nproc-per-node and --grad-accum-steps only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/_args.sh
source "${SCRIPT_DIR}/_args.sh"
# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"
# shellcheck source=scripts/_dist.sh
source "${SCRIPT_DIR}/_dist.sh"

vepopsd_usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -48; }

MODE="${1:-train}"
case "${MODE}" in
  smoke|train|resume) shift || true ;;
  --help|-h) vepopsd_usage; exit 0 ;;
  *) echo "unknown mode '${MODE}' (smoke|train|resume)" >&2; exit 2 ;;
esac

vepopsd_parse_args "model processor training-mode teacher-device eipd-root manifest output-dir \
nproc-per-node global-batch grad-accum-steps max-steps epochs lr max-new-tokens max-answer-tokens \
image-max-pixels lora-r lora-alpha lambda-d alpha adam-beta2 deepspeed workers cpu-threads \
gradient-checkpointing save-steps logging-steps resume smoke-samples subset-samples dry-run \
log-dir run-name" "$@"

# Absolute defaults, same policy as _env.sh: no discovery, override by env or flag.
EIPD_ROOT="${EIPD_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD}"
MODEL="${MODEL:-/nfs/ofs-llm-ssd/user/shengrenren_i/models/Qwen3-VL-8B-Instruct}"
MANIFEST="${MANIFEST:-${EIPD_ROOT}/train.jsonl}"
TRAINING_MODE="${TRAINING_MODE:-lora}"
TEACHER_DEVICE="${TEACHER_DEVICE:-}"
NPROC_PER_NODE="$(vepopsd_nproc)"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
# Future-L1 convention (scripts/run_baseline_future_l1_50k.sh): global batch 128,
# per-device batch 1, grad_accum = global / ranks. EIPD's collator also refuses
# per-device batches above 1, so ranks x accum is the only way to raise the batch.
if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then
  :
else
  DENOM=$((1 * NPROC_PER_NODE))
  if (( GLOBAL_BATCH % DENOM != 0 )); then
    echo "--global-batch ${GLOBAL_BATCH} must be divisible by nproc ${NPROC_PER_NODE}" >&2; exit 2
  fi
  GRAD_ACCUM_STEPS=$((GLOBAL_BATCH / DENOM))
fi
MAX_STEPS="${MAX_STEPS:--1}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-128}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-2097152}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LAMBDA_D="${LAMBDA_D:-0.2}"
ALPHA="${ALPHA:-0.5}"
# --deepspeed maps to $DEEPSPEED in scripts/_args.sh; the internal name is
# DEEPSPEED_CONFIG so the env var and the flag can both be honoured.
[[ -n "${DEEPSPEED:-}" ]] && DEEPSPEED_CONFIG="${DEEPSPEED}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG-${SCRIPT_DIR}/zero2.json}"
WORKERS="${WORKERS:-6}"
CPU_THREADS="${CPU_THREADS:-4}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
RESUME="${RESUME:-}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-32}"
SUBSET_SAMPLES="${SUBSET_SAMPLES:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
RUN_NAME="${RUN_NAME:-eipd-${MODE}}"

case "${TRAINING_MODE}" in
  lora|full) ;;
  *) echo "--training-mode must be lora|full" >&2; exit 2 ;;
esac

# Full finetuning cannot share weights with the frozen teacher, so it needs a
# second 8B copy; LoRA mode disables the adapter instead.
if [[ "${TRAINING_MODE}" == "full" ]]; then
  GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-on}"
  [[ -n "${DEEPSPEED_CONFIG}" ]] || DEEPSPEED_CONFIG="${SCRIPT_DIR}/zero3_optimizer_offload.json"
else
  GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-off}"
fi

cd "${ROOT}"
PY="${VEP_OPSD_ENV_PREFIX}/bin/python"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- preflight ------------------------------------------------------------
[[ -n "${MODEL:-}" && -s "${MODEL}/config.json" ]] || {
  echo "MODEL must be a Qwen3-VL checkpoint directory with config.json (got '${MODEL:-}')" >&2; exit 2; }
[[ -s "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 2; }
[[ -d "${EIPD_ROOT}/frames" ]] || {
  echo "no frames under ${EIPD_ROOT}/frames; run: bash scripts/prepare_eipd_frames.sh extract" >&2; exit 2; }
"${PY}" - "${MODEL}" <<'PY' || exit 2
import json, sys
cfg = json.load(open(sys.argv[1] + "/config.json", encoding="utf-8"))
if cfg.get("model_type") != "qwen3_vl":
    raise SystemExit(f"model_type={cfg.get('model_type')!r}; EIPD needs qwen3_vl")
if cfg.get("latent_id") is not None or cfg.get("max_latent_token") is not None:
    raise SystemExit("this is a Future-L1 checkpoint; the text-token EIPD entry needs the "
                     "original Qwen3-VL checkpoint")
print(f"ok  checkpoint ({cfg['model_type']}, original Qwen3-VL)")
PY

case "${MODE}" in
  train)  GRAD_PROBE_INTERVAL="" ; MAX_STEPS="${MAX_STEPS}" ;;
  resume) GRAD_PROBE_INTERVAL="" ; [[ -z "${RESUME}" ]] && RESUME=last ;;
  smoke)  MAX_STEPS="1" ;;
esac

# Subset manifests must live next to the dataset: EIPD resolves every frame path
# relative to the manifest's own directory.
build_subset() {
  local dst="$1" want="$2" src="$3"
  if [[ -s "${dst}" ]]; then echo "reusing ${dst}"; return 0; fi
  echo "building ${dst} (${want} samples)"
  "${PY}" - "${dst}" "${want}" "${src}" <<'PY'
import json, sys
from pathlib import Path
dst, want, src = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
root = src.parent
kept = []
for line in src.read_text(encoding="utf-8").splitlines():
    if len(kept) >= want:
        break
    row = json.loads(line)
    paths = row["past_files"] + [row["bridge_file"]]
    if all((root / p).is_file() for p in paths):
        kept.append(line)
if len(kept) < want:
    raise SystemExit(f"only {len(kept)}/{want} samples have extracted frames")
dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(f"wrote {len(kept)} samples -> {dst}")
PY
}

if [[ "${MODE}" == "smoke" ]]; then
  build_subset "${EIPD_ROOT}/train.smoke${SMOKE_SAMPLES}.jsonl" "${SMOKE_SAMPLES}" "${MANIFEST}"
  MANIFEST="${EIPD_ROOT}/train.smoke${SMOKE_SAMPLES}.jsonl"
elif [[ "${SUBSET_SAMPLES}" -gt 0 ]]; then
  build_subset "${EIPD_ROOT}/train.subset${SUBSET_SAMPLES}.jsonl" "${SUBSET_SAMPLES}" "${MANIFEST}"
  MANIFEST="${EIPD_ROOT}/train.subset${SUBSET_SAMPLES}.jsonl"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-${HOME}/outputs}/eipd/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ---- trainer arguments ----------------------------------------------------
trainer_args=(
  --model "${MODEL}"
  --manifest "${MANIFEST}"
  --output "${OUTPUT_DIR}"
  --training-mode "${TRAINING_MODE}"
  --epochs "${EPOCHS}"
  --max-steps "${MAX_STEPS}"
  --lr "${LR}"
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-answer-tokens "${MAX_ANSWER_TOKENS}"
  --image-max-pixels "${IMAGE_MAX_PIXELS}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --adam-beta2 "${ADAM_BETA2:-0.95}"
  --lambda-d "${LAMBDA_D}"
  --alpha "${ALPHA}"
  --save-steps "${SAVE_STEPS}"
  --logging-steps "${LOGGING_STEPS}"
  --dataloader-num-workers "${WORKERS}"
  --cpu-threads "${CPU_THREADS}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
)
[[ -n "${PROCESSOR:-}" ]] && trainer_args+=(--processor "${PROCESSOR}")
[[ -n "${TEACHER_DEVICE}" ]] && trainer_args+=(--teacher-device "${TEACHER_DEVICE}")
case "${DEEPSPEED_CONFIG}" in
  ""|none|off|no) ;;
  *) trainer_args+=(--deepspeed "${DEEPSPEED_CONFIG}") ;;
esac
case "${GRADIENT_CHECKPOINTING}" in
  on|true|1) trainer_args+=(--gradient-checkpointing) ;;
  off|false|0) trainer_args+=(--no-gradient-checkpointing) ;;
  *) echo "--gradient-checkpointing must be on|off" >&2; exit 2 ;;
esac
[[ -n "${RESUME}" ]] && trainer_args+=(--resume-from-checkpoint "${RESUME}")
trainer_args+=(${VEPOPSD_PASSTHROUGH[@]+"${VEPOPSD_PASSTHROUGH[@]}"})

# Always go through torchrun, even for one rank: a plain `python -m` on a
# multi-GPU host makes transformers wrap the model in DataParallel, which
# multiplies train_batch_size by n_gpu and breaks the batch-size-1 contract.
vepopsd_dist_args "${NPROC_PER_NODE}"
launcher=("${VEP_OPSD_ENV_PREFIX}/bin/torchrun" "${DIST_ARGS[@]}" -m vep_opsd.eipd_train)

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}-${STAMP}.log"
echo "mode=${MODE} training_mode=${TRAINING_MODE} nproc=${NPROC_PER_NODE} per_device_batch=1 grad_accum=${GRAD_ACCUM_STEPS} global_batch=$((NPROC_PER_NODE * GRAD_ACCUM_STEPS))"
echo "manifest=${MANIFEST}"
echo "model=${MODEL}"
echo "output=${OUTPUT_DIR}"
echo "log=${LOG_FILE}"
printf '%q ' "${launcher[@]}" "${trainer_args[@]}"; echo

if [[ "${DRY_RUN}" == "1" ]]; then echo "(dry run)"; exit 0; fi

"${launcher[@]}" "${trainer_args[@]}" 2>&1 | tee -a "${LOG_FILE}"
