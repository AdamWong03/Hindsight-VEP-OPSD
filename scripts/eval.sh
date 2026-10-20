#!/usr/bin/env bash
# Hindsight VEP-OPSD v1 — FutureBench / TwiFF-Bench generation.
#
#   bash scripts/eval.sh --bench twiff --ckpt <ckpt> --eval-out <dir>
#   bash scripts/eval.sh --bench futurebench --ckpt <ckpt> --eval-out <dir> \
#     --futurebench-json "$FUTUREBENCH_DIR/futurebench_local.json"
#
# Options (defaults in brackets target this machine):
#   --bench twiff|futurebench          required
#   --ckpt DIR                         required: completed VEP-OPSD checkpoint
#   --eval-out DIR                     required: where samples/results are written
#   --task NAME                        lmms-eval task name [twiffbench_future_l1|futurebench_future_l1]
#   --num-processes N                  accelerate processes [2]
#   --batch-size N                     lmms-eval request batch [1]
#   --frames N                         max frames per video [8 twiff | 32 futurebench]
#   --tokens N                         max_new_tokens [4096 twiff | 2048 futurebench]
#   --limit N                          only generate N items (smoke)
#   --system-prompt TEXT               adapter system prompt [You are a helpful assistant.]
#   --eval-port PORT                   accelerate main port [12385]
#   --hf-home DIR                      HF cache [$HF_HOME]
#   --twiff-bench-dir DIR              TwiFF parquet shards [$TWIFF_BENCH_DIR]
#   --futurebench-json FILE            prepared benchmark json [$FUTUREBENCH_DIR/...]
#   --env-name NAME / --env-prefix PATH / --config FILE   as in train_50k.sh
#
# Checkpoints written by vep_opsd/train.py are plain Qwen3-VL directories, so the
# stock lmms-eval `qwen3_vl` adapter is used — no custom model wrapper.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/_args.sh
source "${SCRIPT_DIR}/_args.sh"

vepopsd_usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -30; }

vepopsd_parse_args \
  "bench ckpt eval-out task num-processes batch-size frames tokens limit system-prompt \
eval-port hf-home twiff-bench-dir futurebench-json env-name env-prefix config" "$@"

# Argument validation happens before environment resolution, so a platform run
# fails with the actionable message rather than an env error.
: "${BENCH:?Set --bench twiff or --bench futurebench}"
: "${CKPT:?Set --ckpt to a completed Hindsight VEP-OPSD checkpoint}"
: "${EVAL_OUT:?Set --eval-out to a result directory}"

# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"

case "${BENCH}" in
  twiff)
    TASK="${TASK:-twiffbench_future_l1}"
    FRAMES="${FRAMES:-8}"
    TOKENS="${TOKENS:-4096}" ;;
  futurebench)
    TASK="${TASK:-futurebench_future_l1}"
    FRAMES="${FRAMES:-32}"
    TOKENS="${TOKENS:-2048}" ;;
  *) echo "--bench must be twiff or futurebench" >&2; exit 2 ;;
esac

NUM_PROCESSES="${NUM_PROCESSES:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
HF_HOME="${HF_HOME:-${CACHE_ROOT:-${HOME}/.cache}/hf_eval}"
case "${EVAL_OUT}" in
  /*) ;;
  *) echo "EVAL_OUT must be an absolute path, got '${EVAL_OUT}'" >&2; exit 2 ;;
esac
if ! mkdir -p "${EVAL_OUT}" "${HF_HOME}" 2>/dev/null; then
  echo "cannot create EVAL_OUT='${EVAL_OUT}' or HF_HOME='${HF_HOME}'" >&2
  echo "an empty variable in the job command expands to nothing; use a literal path" >&2
  exit 2
fi

# ---- evaluation framework ---------------------------------------------------
if ! python -c "import lmms_eval" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
lmms_eval is not importable in this environment. Evaluation needs a checkout:

  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval lmms-eval
  bash scripts/setup_env.sh

or point at an existing one with LMMS_EVAL_ROOT=/path/to/lmms-eval
EOF
  exit 2
fi

# ---- benchmark data --------------------------------------------------------
# TwiFF-Bench ships frames inside the parquet shards, so only the shard
# directory is needed. FutureBench resolves absolute `video_path` entries from a
# prepared JSON, which requires fetching the video corpus once.
TASK_DIR="${ROOT}/eval_tasks"
if [[ "${BENCH}" == futurebench ]]; then
  export FUTUREBENCH_JSON="${FUTUREBENCH_JSON:-${FUTUREBENCH_DIR:-${DATA_ROOT:-${HOME}/data}/FutureBench}/futurebench_local.json}"
  if [[ ! -s "${FUTUREBENCH_JSON}" ]]; then
    cat >&2 <<EOF
FutureBench needs a prepared benchmark JSON with absolute video paths.
Build it once (streams the official archives, keeps only referenced clips):

  bash scripts/fetch_futurebench.sh --dest "${FUTUREBENCH_DIR:-${DATA_ROOT:-${HOME}/data}/FutureBench}"

then re-run with --futurebench-json <that dir>/futurebench_local.json
EOF
    exit 2
  fi
  TASK_TEMPLATE="${TASK_DIR}/futurebench/futurebench.yaml.in"
else
  export TWIFF_BENCH_DIR="${TWIFF_BENCH_DIR:-${DATA_ROOT:-${HOME}/data}/TwiFF-Bench}"
  for shard in 00000 00001 00002 00003; do
    [[ -s "${TWIFF_BENCH_DIR}/test-${shard}-of-00004.parquet" ]] || {
      echo "missing TwiFF-Bench shard: ${TWIFF_BENCH_DIR}/test-${shard}-of-00004.parquet" >&2; exit 2; }
  done
  TASK_TEMPLATE="${TASK_DIR}/twiff/twiff.yaml.in"
fi

# lmms-eval reads task YAML verbatim (no env interpolation), so render the
# template in place next to its utils.py.
TASK_YAML="${TASK_TEMPLATE%.in}"
python - "${TASK_TEMPLATE}" "${TASK_YAML}" <<'PY'
import os
import sys
from pathlib import Path

template, target = Path(sys.argv[1]), Path(sys.argv[2])
text = template.read_text(encoding="utf-8")
for key in ("TWIFF_BENCH_DIR", "FUTUREBENCH_JSON"):
    placeholder = f"__{key}__"
    if placeholder in text:
        value = os.environ.get(key)
        if not value:
            raise SystemExit(f"{placeholder} in {template} needs ${key} to be set")
        text = text.replace(placeholder, value)
target.write_text(text, encoding="utf-8")
print(f"rendered {target}")
PY

MODEL_ARGS="pretrained=${CKPT},attn_implementation=sdpa,max_num_frames=${FRAMES},system_prompt=${SYSTEM_PROMPT}"

echo "task:     ${TASK}"
echo "ckpt:     ${CKPT}"
echo "output:   ${EVAL_OUT}"
echo "env:      ${VEP_OPSD_ENV_PREFIX}"
echo "frames:   ${FRAMES}   tokens: ${TOKENS}   procs: ${NUM_PROCESSES}   batch: ${BATCH_SIZE}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME TOKENIZERS_PARALLELISM=false
export TWIFF_SKIP_JUDGE=1   # the answer judge is a separate step: scripts/judge_twiff.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Skip generation when this run already produced samples, so a killed process
# can be restarted without redoing the expensive decode.
if find "${EVAL_OUT}" -name "*_samples_${TASK}.jsonl" -print -quit | read -r _; then
  echo "samples already present under ${EVAL_OUT}; skipping generation"
else
  accelerate launch --num_processes "${NUM_PROCESSES}" \
    --main_process_port="${EVAL_PORT:-12385}" -m lmms_eval \
    --model qwen3_vl \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASK_YAML}" \
    --batch_size "${BATCH_SIZE}" \
    ${LIMIT:+--limit "${LIMIT}"} \
    --gen_kwargs "max_new_tokens=${TOKENS},temperature=0,top_p=1,do_sample=false" \
    --log_samples --verbosity INFO \
    --log_samples_suffix "${TASK}" \
    --output_path "${EVAL_OUT}" \
    2>&1 | tee "${EVAL_OUT}/generation.log"
fi

echo
echo "generation done. next:"
if [[ "${BENCH}" == twiff ]]; then
  echo "  bash scripts/judge_twiff.sh --dir ${EVAL_OUT}"
else
  echo "  python scripts/report_eval_run.py ${EVAL_OUT}"
fi
