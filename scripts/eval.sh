#!/usr/bin/env bash
set -euo pipefail

: "${CKPT:?Set CKPT to a completed Hindsight VEP-OPSD checkpoint}"
: "${EVAL_OUT:?Set EVAL_OUT to a result directory}"
: "${BENCH:?Set BENCH to futurebench or twiff}"

case "${BENCH}" in
  twiff) TASK="${TWIFF_TASK:-twiffbench_future_l1}"; FRAMES=8; TOKENS=4096 ;;
  futurebench) TASK="${FUTUREBENCH_TASK:-futurebench_future_l1}"; FRAMES=32; TOKENS=2048 ;;
  *) echo "BENCH must be futurebench or twiff" >&2; exit 2 ;;
esac

mkdir -p "${EVAL_OUT}"
if [[ -n "${LMMS_EVAL_ROOT:-}" ]]; then
  cd "${LMMS_EVAL_ROOT}"
fi

accelerate launch --num_processes="${EVAL_PROCESSES:-4}" \
  --main_process_port="${EVAL_PORT:-12651}" \
  -m lmms_eval \
  --model qwen3_vl \
  --model_args "pretrained=${CKPT},attn_implementation=sdpa,max_num_frames=${FRAMES},system_prompt=${VEPOPSD_SYSTEM_PROMPT:-You are a helpful assistant.}" \
  --tasks "${TASK}" \
  --batch_size 1 \
  --gen_kwargs "max_new_tokens=${TOKENS},temperature=0,top_p=1,do_sample=false" \
  --log_samples \
  --verbosity INFO \
  --log_samples_suffix hindsight_vep_opsd_v1 \
  --output_path "${EVAL_OUT}" 2>&1 | tee "${EVAL_OUT}/generation.log"

if [[ "${BENCH}" == twiff && "${RUN_TWIFF_JUDGE:-0}" == 1 ]]; then
  SAMPLE_PATH="$(find "${EVAL_OUT}" -name '*_samples_twiffbench_future_l1.jsonl' | sort | tail -n 1)"
  test -s "${SAMPLE_PATH}"
  python tools/twiffbench/post_answer_judge.py \
    --input "${SAMPLE_PATH}" \
    --concurrency "${JUDGE_CONCURRENCY:-16}" \
    --suffix hindsight_vep_opsd_v1 2>&1 | tee "${EVAL_OUT}/answer_judge.log"
fi

