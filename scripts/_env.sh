#!/usr/bin/env bash
# Resolve the reusable micromamba environment and put it on PATH.
#
# Sourced by the train/eval/judge scripts so they behave the same whether they
# are started from an activated shell, from `micromamba run`, or from cron/CI.
#
# Resolution order:
#   1. $VEP_OPSD_ENV_PREFIX     absolute prefix, e.g. /shared/envs/vep (preferred
#                               in job containers, which do not inherit the
#                               developer machine's micromamba envs_dirs)
#   2. $VEP_OPSD_ENV_BASE/<name>  if a shared envs directory is configured
#   3. `micromamba env list` entry named $VEP_OPSD_ENV_NAME
#   4. $HOME/{micromamba,.conda,pixi}/envs/<name> and /opt/conda/envs/<name>
#
# Build it with `bash scripts/setup_env.sh` if it is missing.
VEP_OPSD_ENV_NAME="${VEP_OPSD_ENV_NAME:-vep}"

if [[ -z "${VEP_OPSD_ENV_PREFIX:-}" && -n "${VEP_OPSD_ENV_BASE:-}" ]]; then
  if [[ -x "${VEP_OPSD_ENV_BASE}/${VEP_OPSD_ENV_NAME}/bin/python" ]]; then
    VEP_OPSD_ENV_PREFIX="${VEP_OPSD_ENV_BASE}/${VEP_OPSD_ENV_NAME}"
  fi
fi

if [[ -z "${VEP_OPSD_ENV_PREFIX:-}" ]] && command -v micromamba >/dev/null 2>&1; then
  VEP_OPSD_ENV_PREFIX="$(micromamba env list 2>/dev/null \
    | awk -v name="${VEP_OPSD_ENV_NAME}" '$1 == name {print $NF}' | tail -1)"
fi

if [[ -z "${VEP_OPSD_ENV_PREFIX:-}" ]]; then
  for candidate in \
    "${HOME}/micromamba/envs/${VEP_OPSD_ENV_NAME}" \
    "${HOME}/.conda/envs/${VEP_OPSD_ENV_NAME}" \
    "${HOME}/.pixi/envs/${VEP_OPSD_ENV_NAME}" \
    "/opt/conda/envs/${VEP_OPSD_ENV_NAME}"; do
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

or set the absolute prefix, which is what job containers need because they do not
inherit this machine's micromamba envs_dirs:

  VEP_OPSD_ENV_PREFIX=/path/to/env            # exported, or in .env
  VEP_OPSD_ENV_BASE=/path/to/envs             # directory holding <name>/
EOM
  exit 1
fi

export VEP_OPSD_ENV_PREFIX
export PATH="${VEP_OPSD_ENV_PREFIX}/bin:${PATH}"
