#!/usr/bin/env bash
# Shared distributed launcher details for the EIPD scripts.
#
# Card count defaults to every visible GPU. Multi-node variables injected by a
# cluster (DISTRIBUTED_WORLD_SIZE / DISTRIBUTED_NODE_RANK / MASTER_ADDR /
# MASTER_PORT) are honoured.
#
# Always drives <env>/bin/torchrun by absolute path: inside a job container the
# environment may not be "activated" in the login-shell sense, and a plain
# `python -m` start would let transformers wrap the model in DataParallel, which
# doubles the batch and trips the batch-size-1 collator.
#
# NOTE: `vepopsd_dist_args` fills the DIST_ARGS **array**. Earlier this returned
# newline-separated text, which silently produced the single argument
# "--nproc_per_node 8" and made torchrun fail with "unknown option".

vepopsd_nproc() {
    if [[ -n "${NPROC_PER_NODE:-}" ]]; then
        printf '%s\n' "${NPROC_PER_NODE}"
        return
    fi
    "${VEP_OPSD_ENV_PREFIX}/bin/python" - <<'PY'
import torch
print(max(1, torch.cuda.device_count()))
PY
}

vepopsd_nnodes() { printf '%s\n' "${NNODES:-${DISTRIBUTED_WORLD_SIZE:-1}}"; }
vepopsd_node_rank() { printf '%s\n' "${NODE_RANK:-${DISTRIBUTED_NODE_RANK:-0}}"; }

# Fills the caller's DIST_ARGS array with one argv element per flag/value.
vepopsd_dist_args() {
    local nproc="${1:-$(vepopsd_nproc)}"
    DIST_ARGS=(
        --nproc_per_node "${nproc}"
        --nnodes "$(vepopsd_nnodes)"
        --node_rank "$(vepopsd_node_rank)"
        --master_addr "${MASTER_ADDR:-127.0.0.1}"
        --master_port "${MASTER_PORT:-29671}"
    )
}
