#!/usr/bin/env bash
# EIPD-Data-v1 — unpack the published frame shards and verify them.
#
# The dataset ships the manifests as files but the frames only as tar shards:
#
#   EIPD/
#     train.jsonl  val.jsonl  dataset_summary.json
#     archives/train-00000-of-00025.tar ... val-00004-of-00005.tar   (111 GB)
#
# vep_opsd.eipd_train.ManifestDataset resolves every manifest path relative to
# the directory that holds the manifest, so the shards must be extracted into
# EIPD/ itself (they already contain the frames/ prefix).
#
#   bash scripts/prepare_eipd_frames.sh extract            # unpack everything
#   bash scripts/prepare_eipd_frames.sh verify             # hash every frame
#   bash scripts/prepare_eipd_frames.sh status             # cheap progress check
#
# Options (everything else is forwarded to nothing; this script has no trainer):
#   --eipd-root DIR    dataset root [$EIPD_ROOT, then $DATA_ROOT/EIPD]
#   --jobs N           parallel tar processes [4]
#   --split NAME       train | val | all [all]
#
# Extraction is restartable: shards whose frames already exist are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/_args.sh
source "${SCRIPT_DIR}/_args.sh"
# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"

vepopsd_usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -30; }

if [[ $# -gt 0 && "$1" == "--help" ]]; then vepopsd_usage; exit 0; fi

ACTION="${1:-extract}"
shift || true
vepopsd_parse_args "eipd-root jobs split" "$@"

EIPD_ROOT="${EIPD_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/data/EIPD}"
JOBS="${JOBS:-4}"
SPLIT="${SPLIT:-all}"
case "${ACTION}" in
  extract|verify|status) ;;
  *) echo "unknown action '${ACTION}' (extract|verify|status)" >&2; exit 2 ;;
esac
[[ -n "${EIPD_ROOT}" && "${EIPD_ROOT}" == /* ]] || {
  echo "EIPD_ROOT must be an absolute path, got '${EIPD_ROOT}'" >&2; exit 2; }

cd "${ROOT}"
PY="${VEP_OPSD_ENV_PREFIX}/bin/python"

manifest_count() { wc -l < "$1"; }

case "${ACTION}" in
  status)
    for split in train val; do
      manifest="${EIPD_ROOT}/${split}.jsonl"
      [[ -s "${manifest}" ]] || { echo "missing ${manifest}" >&2; continue; }
      have=$( { find "${EIPD_ROOT}/frames/${split}" -maxdepth 2 -name '*.png' 2>/dev/null || true; } | wc -l )
      want=$("${PY}" - "${manifest}" <<'PY'
import json, sys
n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    n += len(row["past_files"]) + len(row["future_files"]) + 1
print(n)
PY
)
      echo "${split}: frames ${have}/${want} samples $(manifest_count "${manifest}")"
    done
    du -sh "${EIPD_ROOT}/archives" "${EIPD_ROOT}/frames" 2>/dev/null || true
    ;;

  extract)
    case "${SPLIT}" in train|val) shards=("${EIPD_ROOT}"/archives/"${SPLIT}"-*.tar) ;;
                    all)          shards=("${EIPD_ROOT}"/archives/*.tar) ;;
                    *) echo "--split must be train|val|all" >&2; exit 2 ;; esac
    [[ -e "${shards[0]}" ]] || { echo "no shards under ${EIPD_ROOT}/archives" >&2; exit 2; }
    echo "extracting ${#shards[@]} shard(s) into ${EIPD_ROOT} with ${JOBS} jobs"
    export EIPD_ROOT
    printf '%s\0' "${shards[@]}" | xargs -0 -P "${JOBS}" -I{} bash -c '
      set -euo pipefail
      shard="$1"
      tar -xf "$shard" -C "${EIPD_ROOT}"
      echo "done $(basename "$shard")"
    ' _ {}
    echo "extract complete"
    "${BASH_SOURCE[0]}" status --eipd-root "${EIPD_ROOT}"
    ;;

  verify)
    # eipd_final_data verify re-hashes both manifests, checks bridge chronology,
    # source uniqueness and the SHA-256 of every past/bridge/future PNG.
    "${PY}" -m vep_opsd.eipd_final_data verify --output "${EIPD_ROOT}"
    ;;
esac
