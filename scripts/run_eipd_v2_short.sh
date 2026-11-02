#!/usr/bin/env bash
# EIPD — short-chain pipeline (4 latent steps).
#
#   bash scripts/run_eipd_v2_short.sh            # Stage 1 + Stage 2 (gain-selected KD)
#   bash scripts/run_eipd_v2_short.sh gain       # same
#   bash scripts/run_eipd_v2_short.sh js         # Stage 2 with the original JS weighting
#   bash scripts/run_eipd_v2_short.sh pure       # Stage 1 only (pure-latent control, no KD)
#   DRY_RUN=1 bash scripts/run_eipd_v2_short.sh  # print both commands, run nothing
#
# What makes this "short":
#   * K=4 latent steps. With K=16 the last anchor is always k=16 while 69.4% of the
#     samples have a single future frame, so only ~8.6% of latent steps ever receive
#     direct visual supervision; at K=4 that becomes 25% (m=1) to 100% (m=4).
#     Frames are never interpolated: when m > K the dataset keeps K real frames.
#   * --eta 0          no temporal term (identically zero for 69.4% of samples).
#   * --single-view 1  one view per sample (Past or Past+Bridge, at random). Same
#     expectation as (CE_p + CE_+)/2 with half the prefills and answer passes; the
#     future loss is doubled for past-only samples so the objective's scale is kept.
#   * 2 / 1 epochs     one full pass of future-evidence distillation, not more epochs
#     by default.
#
# The 16-step 3-epoch run stays as the control arm; nothing here touches its output.
set -euo pipefail

VARIANT="${1:-gain}"
case "${VARIANT}" in
  gain|js|pure) ;;
  *) echo "usage: $0 [gain|js|pure]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Machine paths: absolute defaults, same policy as _env.sh. Job containers do not
# inherit this machine's .env or its micromamba envs_dirs.
EIPD_ROOT="${EIPD_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD}"
MODEL="${MODEL:-/nfs/ofs-llm-ssd/user/shengrenren_i/models/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/outputs}"

# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"
# shellcheck source=scripts/_dist.sh
source "${SCRIPT_DIR}/_dist.sh"

PY="${VEP_OPSD_ENV_PREFIX}/bin/python"
NPROC="$(vepopsd_nproc)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_BASE="${OUT_BASE:-${OUTPUT_ROOT}/eipd/v2-short-${VARIANT}-${STAMP}}"
MANIFEST="${EIPD_ROOT}/train.jsonl"
FEATURES="${EIPD_ROOT}/future_features/train"

LATENT_STEPS="${LATENT_STEPS:-4}"
ETA="${ETA:-0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
SINGLE_VIEW="${SINGLE_VIEW:-1}"
SAVE_STEPS="${SAVE_STEPS:-200}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUT_BASE}" "${ROOT}/logs"

say() { echo; echo "===== $* ====="; }

say "plan"
echo "variant   : ${VARIANT}"
echo "latent    : K=${LATENT_STEPS}, eta=${ETA}, single-view=${SINGLE_VIEW}"
echo "epochs    : stage1=${STAGE1_EPOCHS}, stage2=${STAGE2_EPOCHS}$([[ ${VARIANT} == pure ]] && echo ' (stage2 skipped)')"
echo "nproc     : ${NPROC}"
echo "model     : ${MODEL}"
echo "eipd_root : ${EIPD_ROOT}"
echo "out       : ${OUT_BASE}"

[[ -s "${MODEL}/config.json" ]] || { echo "missing model: ${MODEL}" >&2; exit 2; }
[[ -s "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 2; }
if ! compgen -G "${FEATURES}/shard_*.pt" >/dev/null; then
  echo "future-feature cache missing under ${FEATURES}" >&2
  echo "build it once with: ${PY} scripts/precompute_future_features.py --help" >&2
  exit 2
fi

dry=()
[[ "${DRY_RUN}" == "1" ]] && dry=(--dry-run 1)

say "stage 1 (K=${LATENT_STEPS}, single view, ${STAGE1_EPOCHS} epochs)"
bash "${SCRIPT_DIR}/train_eipd_stage1.sh" train \
  --model "${MODEL}" --eipd-root "${EIPD_ROOT}" \
  --manifest "${MANIFEST}" --features "${FEATURES}" \
  --output-dir "${OUT_BASE}/stage1" --nproc-per-node "${NPROC}" \
  --training-mode "${TRAINING_MODE:-full}" \
  --latent-steps "${LATENT_STEPS}" --eta "${ETA}" \
  --epochs "${STAGE1_EPOCHS}" --single-view "${SINGLE_VIEW}" \
  --save-steps "${SAVE_STEPS}" "${dry[@]}"

if [[ "${VARIANT}" == "pure" ]]; then
  say "done — pure-latent arm only"
  echo "stage 1 : ${OUT_BASE}/stage1"
  echo "Use this as the no-distillation control: same model, no future-evidence KD."
  exit 0
fi

say "stage 2 (K=${LATENT_STEPS}, ${STAGE2_EPOCHS} epoch, kd-mode=${VARIANT})"
bash "${SCRIPT_DIR}/train_eipd_stage2.sh" train \
  --stage1 "${OUT_BASE}/stage1" --base-model "${MODEL}" \
  --eipd-root "${EIPD_ROOT}" --manifest "${MANIFEST}" --features "${FEATURES}" \
  --output-dir "${OUT_BASE}/stage2" --nproc-per-node "${NPROC}" \
  --latent-steps "${LATENT_STEPS}" --epochs "${STAGE2_EPOCHS}" \
  --eta "${ETA}" --kd-mode "${VARIANT}" "${dry[@]}"

say "done"
cat <<EOF
stage 1 : ${OUT_BASE}/stage1
stage 2 : ${OUT_BASE}/stage2
logs    : ${ROOT}/logs/stage{1,2}-*.log

Read, in this order:
  stage1  state_loss (must fall; with K=16 it started at 0.97)
          ce_past / ce_bridge (only one of them is non-zero per sample: single view)
          view_bridge (fraction of Bridge views actually taken, should hover near 0.5)
          nonfinite_total (must stay 0)
  stage2  eipd_loss, kd_active (share of samples whose Bridge improved the gold
          likelihood), gold_logprob_gain, teacher_logit_drift (must stay ~0)

If this arm looks good, the clean 4-vs-16 comparison needs a 16-step arm with the
same epoch count: --latent-steps 16 --eta 0 --epochs 2 --single-view 1.
EOF
