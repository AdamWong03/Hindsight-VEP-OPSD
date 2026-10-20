#!/usr/bin/env bash
# Score generated TwiFF samples with the answer-only GPT judge, then report.
#
# The judge is an OpenAI-compatible server configured in lmms-eval/.env
# (OPENAI_API_BASE / LOCAL_LLM), normally hosted on the machine that runs this
# script — so run it where that server is reachable, after generation finished
# anywhere.
#
#   bash scripts/judge_twiff.sh --dir <run>-eval/twiff
#   bash scripts/judge_twiff.sh --ckpt <run>
#
# Already-scored runs are reported, not rescored. Re-score on purpose with
# --force 1 (the judge writes its own suffixed files; nothing is overwritten).
#
# Options:
#   --dir DIR            evaluation directory containing *_samples_<task>.jsonl
#   --ckpt DIR           checkpoint directory; implies <output-root>/<dir>-eval
#   --output-root DIR    base for --ckpt [$OUTPUT_ROOT/hindsight-vep-opsd]
#   --task NAME          lmms-eval task [twiffbench_future_l1]
#   --suffix NAME        judge file suffix [vepopsd]
#   --concurrency N      in-flight judge requests [16]
#   --force 1            rescore even when a summary exists [0]
#   --env-name NAME / --env-prefix PATH / --config FILE   as in train_50k.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/_args.sh
source "${SCRIPT_DIR}/_args.sh"

vepopsd_usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -22; }

vepopsd_parse_args \
  "dir ckpt output-root task suffix concurrency force env-name env-prefix config" "$@"

OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_ROOT_BASE:-${HOME}/outputs}/hindsight-vep-opsd}"
TASK="${TASK:-twiffbench_future_l1}"
JUDGE_SUFFIX="${JUDGE_SUFFIX:-vepopsd}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
FORCE="${FORCE:-0}"

if [[ -z "${DIR:-}" ]]; then
  [[ -n "${CKPT:-}" ]] || { echo "either --dir or --ckpt is required" >&2; exit 2; }
  DIR="${OUTPUT_ROOT}/$(basename "${CKPT}")-eval"
fi
[[ -d "${DIR}" ]] || { echo "not a directory: ${DIR}" >&2; exit 2; }

# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"

PY="${VEP_OPSD_ENV_PREFIX}/bin/python"

SAMPLES="$(find "${DIR}" -name "*_samples_${TASK}.jsonl" -print 2>/dev/null | sort | tail -1)"
[[ -s "${SAMPLES}" ]] || { echo "no generation found under ${DIR}; run scripts/eval.sh first" >&2; exit 1; }
echo "samples: ${SAMPLES}"

# The judge loads lmms-eval/.env itself; probe the server first so a wrong host
# fails here instead of after a partial scoring pass.
"${PY}" - "${ROOT}/lmms-eval/.env" <<'PY'
import json
import os
import sys
import urllib.request
from pathlib import Path

env_path = Path(sys.argv[1])
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
base = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
model = os.environ.get("LOCAL_LLM") or os.environ.get("JUDGE_MODEL") or "gpt-4o"
request = urllib.request.Request(
    f"{base}/models",
    headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"},
)
try:
    with urllib.request.urlopen(request, timeout=15) as response:
        ids = [m.get("id") for m in json.loads(response.read()).get("data", [])]
except Exception as exc:
    raise SystemExit(f"judge server unreachable at {base}: {type(exc).__name__}: {exc}")
print(f"judge server ok: {base} (model={model}, advertised={ids[:3]})")
PY

if [[ "${FORCE}" != "1" ]] && find "${DIR}" -name "*.answer_judge.${JUDGE_SUFFIX}.summary.json" -print -quit | read -r _; then
  echo "already scored with suffix ${JUDGE_SUFFIX}; reporting the existing records"
else
  ( cd "${ROOT}/lmms-eval" && "${PY}" tools/twiffbench/post_answer_judge.py \
      --input "${SAMPLES}" \
      --concurrency "${JUDGE_CONCURRENCY}" \
      --suffix "${JUDGE_SUFFIX}" 2>&1 | tee "${DIR}/answer_judge.log" )
fi

"${PY}" "${ROOT}/scripts/report_eval_run.py" "${DIR}"
