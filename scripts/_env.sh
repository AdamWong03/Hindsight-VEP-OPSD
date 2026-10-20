#!/usr/bin/env bash
# Resolve the reusable micromamba environment and put it on PATH.
#
# Sourced by the train/eval/judge scripts so they behave the same whether they
# are started from an activated shell, from `micromamba run`, or from cron/CI.
#
# Resolution order:
#   1. $VEP_OPSD_ENV_PREFIX                          (explicit override)
#   2. `micromamba env list` entry named $VEP_OPSD_ENV_NAME
#   3. the usual envs_dirs locations for that name
#
# Build it with `bash scripts/setup_env.sh` if it is missing.
VEP_OPSD_ENV_NAME="${VEP_OPSD_ENV_NAME:-vep}"

if [[ -z "${VEP_OPSD_ENV_PREFIX:-}" ]] && command -v micromamba >/dev/null 2>&1; then
  VEP_OPSD_ENV_PREFIX="$(micromamba env list 2>/dev/null \
    | awk -v name="${VEP_OPSD_ENV_NAME}" '$1 == name {print $NF}' | tail -1)"
fi

if [[ -z "${VEP_OPSD_ENV_PREFIX:-}" ]]; then
  for candidate in \
    "${HOME}/micromamba/envs/${VEP_OPSD_ENV_NAME}" \
    "${HOME}/.conda/envs/${VEP_OPSD_ENV_NAME}"; do
    if [[ -x "${candidate}/bin/python" ]]; then
      VEP_OPSD_ENV_PREFIX="${candidate}"
      break
    fi
  done
fi

if [[ -z "${VEP_OPSD_ENV_PREFIX:-}" || ! -x "${VEP_OPSD_ENV_PREFIX}/bin/python" ]]; then
  cat >&2 <<EOM
micromamba environment '${VEP_OPSD_ENV_NAME}' not found.
Create it once with:

  bash scripts/setup_env.sh

or point at an existing prefix with VEP_OPSD_ENV_PREFIX=/path/to/env
EOM
  exit 1
fi

export VEP_OPSD_ENV_PREFIX
export PATH="${VEP_OPSD_ENV_PREFIX}/bin:${PATH}"
