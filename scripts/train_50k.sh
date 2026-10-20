#!/usr/bin/env bash
# Hindsight-Privileged VEP-OPSD — 50K LoRA training.
#
# Configuration is command-line only: no `VAR=value` prefix, no exported env var.
#
#   bash scripts/train_50k.sh --output-dir "$OUTPUT_ROOT/run1" \
#     --nproc-per-node 8 --global-batch-size 128
#
# Every option also still works as an environment variable; an explicit flag
# wins. Unrecognised flags are forwarded to the trainer (HF arguments), e.g.
#   --image_min_pixels 4096 --freeze_vision_tower False
#
# Paths come from .env (see .env.example) or from these flags; MODEL and DATA
# have no built-in default on purpose.
#
# Options (defaults in brackets):
#   --model DIR                Qwen3-VL checkpoint [$MODEL, required]
#   --data FILE                VEP manifest [$DATA, required]
#   --video-root DIR           root for relative manifest paths [off: absolute paths]
#   --output-dir DIR           checkpoints/logs/rollouts [$OUTPUT_ROOT/hindsight-vep-opsd/<ts>]
#   --log-dir DIR              tee destination [<repo>/logs]
#   --run-name NAME            run label [vep-opsd-50k-<basename output dir>]
#   --nproc-per-node N         torchrun processes [8]
#   --global-batch-size N      optimizer batch [128]
#   --per-device-batch-size N  micro batch per rank [1]
#   --grad-accum-steps N       override the derived accumulation [derived]
#   --max-steps N              stop after N optimizer steps [HF default]
#   --epochs N                 epochs [1]
#   --lr F                     learning rate [2e-5]
#   --max-new-tokens N         rollout length [512]
#   --distillation-top-k N     Teacher support size per generated token [100]
#   --distillation-temperature F  token-distribution temperature [1.0]
#   --jsd-beta F               Teacher weight in generalized JSD [0.5]
#   --lora-r N                 Student LoRA rank [64]
#   --lora-alpha N             Student LoRA alpha [128]
#   --save-steps N             checkpoint interval [50]
#   --save-total-limit N       kept checkpoints [3]
#   --logging-steps N          log interval [1]
#   --dataloader-num-workers N [4]
#   --grad-check BOOL          gradient checkpointing [True]
#   --deepspeed-config FILE    ZeRO config [scripts/zero2.json]
#   --report-to TARGET         none|wandb|tensorboard [none]
#   --wandb-mode MODE          offline|online [offline]
#   --master-port PORT         torchrun rendezvous port [29551]
#   --seed N                   [42]
#   --ddp-timeout SECONDS      collective timeout; the first step pays model load [1800]
#                              plus the dataloader warm-up, so 10 min is too tight
#   --env-name NAME            micromamba env [vep]
#   --env-prefix PATH          micromamba prefix [auto-detected]
#   --config FILE              shell file of defaults; flags override it
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/_args.sh
source "${SCRIPT_DIR}/_args.sh"

vepopsd_usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -45; }

vepopsd_parse_args \
  "model data video-root output-dir log-dir run-name nproc-per-node global-batch-size \
per-device-batch-size grad-accum-steps max-steps epochs lr max-new-tokens \
distillation-top-k distillation-temperature jsd-beta lora-r lora-alpha save-steps \
save-total-limit logging-steps dataloader-num-workers grad-check deepspeed-config \
report-to wandb-mode master-port seed ddp-timeout env-name env-prefix" "$@"

# ---- environment -----------------------------------------------------------
# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "${ROOT}"

# ---- paths -----------------------------------------------------------------
: "${MODEL:?Set --model or MODEL to the Qwen3-VL checkpoint directory (see .env.example)}"
: "${DATA:?Set --data or DATA to the training manifest (see .env.example)}"
# Manifests with absolute video paths need no root; --video-root is only a
# fallback for relative ones.
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-${HOME}/outputs}/hindsight-vep-opsd/$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"

for path in "${MODEL}/config.json" "${DATA}"; do
  [[ -e "${path}" ]] || { echo "missing: ${path}" >&2; exit 2; }
done

# A job command is expanded by the submitter's shell, which does not read .env:
# `--output-dir $OUTPUT_ROOT/...` silently becomes `/...` when OUTPUT_ROOT is
# unset there. Fail with the reason instead of trying to write to /.
case "${OUTPUT_DIR}" in
  /*) ;;
  *) echo "OUTPUT_DIR must be an absolute path, got '${OUTPUT_DIR}'" >&2; exit 2 ;;
esac
if ! mkdir -p "${OUTPUT_DIR}" 2>/dev/null; then
  cat >&2 <<EOM
cannot create OUTPUT_DIR='${OUTPUT_DIR}'
If that path lost a leading directory, a variable in the job command expanded to
nothing -- the submitting shell does not read .env. Either pass a literal path,
or omit --output-dir and let .env's OUTPUT_ROOT supply it (currently
'${OUTPUT_ROOT:-<unset>}').
EOM
  exit 2
fi
mkdir -p "${LOG_DIR}"

# ---- parallelism -----------------------------------------------------------
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
DENOM=$(( NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE ))
if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then
  GRAD_ACCUM="${GRAD_ACCUM_STEPS}"
elif (( GLOBAL_BATCH_SIZE % DENOM == 0 )); then
  GRAD_ACCUM=$(( GLOBAL_BATCH_SIZE / DENOM ))
else
  cat >&2 <<EOF
GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by
nproc-per-node * per-device-batch-size (${DENOM}); pass --grad-accum-steps to
set the accumulation directly.
EOF
  exit 2
fi

# ---- run identity ----------------------------------------------------------
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-Hindsight-VEP-OPSD}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${OUTPUT_DIR}"
# decord/opencv frame-reader noise
export OPENCV_FFMPEG_LOGLEVEL=-8
export AV_LOG_LEVEL=quiet
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

RUN_NAME="${RUN_NAME:-vep-opsd-50k-$(basename "${OUTPUT_DIR}")}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
echo "run:     ${RUN_NAME}"
echo "env:     ${VEP_OPSD_ENV_PREFIX}"
echo "output:  ${OUTPUT_DIR}"
echo "log:     ${LOG_DIR}/${RUN_NAME}-${TIMESTAMP}.log"
echo "gpus:    ${NPROC_PER_NODE} x per_device=${PER_DEVICE_BATCH_SIZE} -> grad_accum=${GRAD_ACCUM} (global ${GLOBAL_BATCH_SIZE})"

# Provenance for platform runs where the submit command is not visible in here.
{
  echo "run_name=${RUN_NAME}"
  echo "model=${MODEL}"
  echo "data=${DATA}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
  echo "grad_accum_steps=${GRAD_ACCUM}"
  echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
  echo "env_prefix=${VEP_OPSD_ENV_PREFIX}"
  echo "passthrough=$(printf '%q ' ${VEPOPSD_PASSTHROUGH[@]+"${VEPOPSD_PASSTHROUGH[@]}"})"
} > "${OUTPUT_DIR}/launch.txt"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT:-29551}" \
  -m vep_opsd.train \
  --run_name "${RUN_NAME}" \
  --model_name_or_path "${MODEL}" \
  --manifest "${DATA}" \
  ${VIDEO_ROOT:+--video_root "${VIDEO_ROOT}"} \
  --output_dir "${OUTPUT_DIR}" \
  --deepspeed "${DEEPSPEED_CONFIG:-scripts/zero2.json}" \
  --bf16 True \
  --fp16 False \
  --tf32 False \
  --gradient_checkpointing "${GRAD_CHECK:-True}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  ${MAX_STEPS:+--max_steps "${MAX_STEPS}"} \
  --num_train_epochs "${EPOCHS:-1}" \
  --learning_rate "${LR:-2e-5}" \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-50}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --remove_unused_columns False \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --dataloader_pin_memory False \
  --ddp_find_unused_parameters False \
  --report_to "${REPORT_TO:-none}" \
  --seed "${SEED:-42}" \
  --ddp_timeout "${DDP_TIMEOUT:-1800}" \
  --sampling_temperature 0.8 \
  --sampling_top_p 0.95 \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  --distillation_top_k "${DISTILLATION_TOP_K:-100}" \
  --distillation_temperature "${DISTILLATION_TEMPERATURE:-1.0}" \
  --jsd_beta "${JSD_BETA:-0.5}" \
  --add_tail_bucket True \
  --lora_r "${LORA_R:-64}" \
  --lora_alpha "${LORA_ALPHA:-128}" \
  ${VEPOPSD_PASSTHROUGH[@]+"${VEPOPSD_PASSTHROUGH[@]}"} \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}-${TIMESTAMP}.log"
