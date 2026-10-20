#!/usr/bin/env bash
# Create (or update) the reusable micromamba environment for Hindsight VEP-OPSD.
#
#   bash scripts/setup_env.sh              # create/refresh the env
#   bash scripts/setup_env.sh --env-name vep --skip-pypi 1
#
# Options:
#   --env-name NAME     micromamba environment name [vep]
#   --index URL         PyPI index [unset: pip's default]
#   --host HOST         allowed insecure index host [unset]
#   --skip-pypi 1       only build the conda layer
#   --config FILE       shell file of defaults
#
# The env is created by name, so it lands in micromamba's shared envs_dirs and
# can be reused by any other checkout on this machine:
#
#   micromamba activate vep
#   eval "$(micromamba shell hook -s bash)"
#
# The PyPI layer uses uv when available (it shares the wheel cache with the
# other project envs) and falls back to pip.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/_args.sh
source "${SCRIPT_DIR}/_args.sh"

vepopsd_usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -18; }

vepopsd_parse_args "env-name index host skip-pypi config" "$@"

ENV_NAME="${ENV_NAME:-vep}"
# Optional mirror, e.g. for an air-gapped cluster: --index URL --host HOST, or
# PYPI_INDEX/PYPI_HOST from .env. Empty means pip/uv use their default index.
PYPI_INDEX="${PYPI_INDEX:-}"
PYPI_HOST="${PYPI_HOST:-}"
INDEX_ARGS=()
if [[ -n "${PYPI_INDEX}" ]]; then
  INDEX_ARGS+=(--index-url "${PYPI_INDEX}")
  [[ -n "${PYPI_HOST}" ]] && INDEX_ARGS+=(--allow-insecure-host "${PYPI_HOST}")
fi

command -v micromamba >/dev/null 2>&1 || { echo "micromamba not found on PATH" >&2; exit 1; }

# ---- conda layer -----------------------------------------------------------
if micromamba env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "conda env '${ENV_NAME}' already exists"
else
  echo "creating conda env '${ENV_NAME}' (python 3.12 + ffmpeg)"
  micromamba create -n "${ENV_NAME}" -c conda-forge python=3.12 ffmpeg pip -y
fi

ENV_PREFIX="$(micromamba env list | awk -v name="${ENV_NAME}" '$1 == name {print $NF}' | tail -1)"
[[ -n "${ENV_PREFIX}" && -x "${ENV_PREFIX}/bin/python" ]] || {
  echo "could not resolve env prefix for '${ENV_NAME}'" >&2; exit 1; }
echo "env prefix: ${ENV_PREFIX}"

[[ "${SKIP_PYPI:-0}" == "1" ]] && exit 0

# ---- PyPI layer ------------------------------------------------------------
cd "${ROOT}"

# The evaluation framework is a separate checkout (not vendored in git). Widen
# its exact dependency pins first, or installing it would downgrade numpy and
# datasets under the training stack.
EDITABLE=(-e .)
if [[ -d "${LMMS_EVAL_ROOT:-lmms-eval}" ]]; then
  LMMS_EVAL_DIR="${LMMS_EVAL_ROOT:-lmms-eval}"
  "${ENV_PREFIX}/bin/python" scripts/patch_lmms_eval.py "${LMMS_EVAL_DIR}"
  EDITABLE+=(-e "${LMMS_EVAL_DIR}")
else
  cat <<'EOF'
note: no lmms-eval checkout found, so evaluation will not be installed.
      Training works without it; to add evaluation later:

        git clone https://github.com/EvolvingLMMs-Lab/lmms-eval lmms-eval
        bash scripts/setup_env.sh

      or point at an existing checkout with LMMS_EVAL_ROOT=/path/to/lmms-eval
EOF
fi

if command -v uv >/dev/null 2>&1; then
  echo "installing requirements.txt with uv"
  uv pip install --python "${ENV_PREFIX}/bin/python" -r requirements.txt \
    ${INDEX_ARGS[@]+"${INDEX_ARGS[@]}"}
  echo "installing editable packages: ${EDITABLE[*]}"
  uv pip install --python "${ENV_PREFIX}/bin/python" "${EDITABLE[@]}" \
    ${INDEX_ARGS[@]+"${INDEX_ARGS[@]}"}
else
  echo "installing requirements.txt with pip"
  "${ENV_PREFIX}/bin/python" -m pip install -r requirements.txt
  "${ENV_PREFIX}/bin/python" -m pip install "${EDITABLE[@]}"
fi

# ---- report ----------------------------------------------------------------
"${ENV_PREFIX}/bin/python" - <<'PY'
import importlib.metadata as md

for name in ("torch", "transformers", "accelerate", "deepspeed", "numpy",
             "decord", "qwen-vl-utils", "lmms-eval", "hindsight-vep-opsd"):
    try:
        print(f"  {name}=={md.version(name)}")
    except Exception:
        print(f"  {name}: MISSING")
PY
echo
echo "activate with:  micromamba activate ${ENV_NAME}"
echo "smoke train:    micromamba run -n ${ENV_NAME} bash scripts/train_smoke.sh"
