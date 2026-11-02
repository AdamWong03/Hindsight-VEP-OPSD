#!/usr/bin/env bash
# The training environment, by absolute path — no discovery.
#
# Job containers do not inherit this machine's micromamba envs_dirs, so looking the
# env up by name silently fails there. The path is fixed and overridable:
#
#   VEP_OPSD_ENV_PREFIX=/path/to/env bash scripts/train_eipd_stage1.sh ...
#
# Sourced by the train/eval/judge scripts.
VEP_OPSD_ENV_PREFIX="${VEP_OPSD_ENV_PREFIX:-/nfs/ofs-llm-ssd/user/shengrenren_i/envs/vep}"

if [[ ! -x "${VEP_OPSD_ENV_PREFIX}/bin/python" ]]; then
    echo "missing environment: ${VEP_OPSD_ENV_PREFIX}/bin/python" >&2
    echo "override with VEP_OPSD_ENV_PREFIX=/path/to/env" >&2
    exit 1
fi
export VEP_OPSD_ENV_PREFIX

# Activate through the shell hook so activate.d scripts (CUDA paths, LD_LIBRARY_PATH)
# run; the PATH export below keeps torchrun/python resolvable either way.
if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook -s bash 2>/dev/null)" || true
    micromamba activate "${VEP_OPSD_ENV_PREFIX}" >/dev/null 2>&1 \
        && echo "env: micromamba activate ${VEP_OPSD_ENV_PREFIX}" >&2
elif command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${base}/etc/profile.d/conda.sh"
        conda activate "${VEP_OPSD_ENV_PREFIX}" >/dev/null 2>&1 \
            && echo "env: conda activate ${VEP_OPSD_ENV_PREFIX}" >&2
    fi
fi

export PATH="${VEP_OPSD_ENV_PREFIX}/bin:${PATH}"
