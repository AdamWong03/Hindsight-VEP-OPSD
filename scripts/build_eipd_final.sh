#!/usr/bin/env bash
# EIPD-Data-v1 stage launcher. Requires verified video provenance.
set -eo pipefail
stage="$1"
if [[ -z "$EIPD_ROOT" ]]; then echo "Set EIPD_ROOT=/data/EIPD-Data-v1" >&2; exit 2; fi
py="python"
if [[ -n "$PYTHON" ]]; then py="$PYTHON"; fi
case "$stage" in
  plan)
    if [[ -z "$EXCLUDE_FUTUREL1" || -z "$EXCLUDE_FUTUREBENCH" || -z "$EXCLUDE_TWIFFBENCH" ]]; then
      echo "Set all three verified EXCLUDE_* source video lists" >&2; exit 2
    fi
    if [[ -n "$TWIFF_META_JSONL" ]]; then
      "$py" -m vep_opsd.eipd_final_data plan --input "$TWIFF_META_JSONL" \
        --output "$EIPD_ROOT" --exclusions "$EXCLUDE_FUTUREL1" "$EXCLUDE_FUTUREBENCH" "$EXCLUDE_TWIFFBENCH"
    else
      "$py" -m vep_opsd.eipd_final_data plan --hf-dataset Liu-Junhua/TwiFF-2.7M \
        --output "$EIPD_ROOT" --exclusions "$EXCLUDE_FUTUREL1" "$EXCLUDE_FUTUREBENCH" "$EXCLUDE_TWIFFBENCH"
    fi
    ;;
  panda-subset)
    if [[ -z "$PANDA_CSV" ]]; then echo "Set PANDA_CSV" >&2; exit 2; fi
    "$py" -m vep_opsd.eipd_final_data panda-subset \
      --csv "$PANDA_CSV" --video-ids "$EIPD_ROOT/candidate_source_ids.txt" \
      --output-csv "$EIPD_ROOT/panda_sources.csv"
    ;;
  materialize)
    if [[ -z "$VIDEO_ROOT" ]]; then echo "Set VIDEO_ROOT to verified TwiFF clips" >&2; exit 2; fi
    if [[ -n "$VIDEO_MAP" ]]; then
      "$py" -m vep_opsd.eipd_final_data materialize --output "$EIPD_ROOT" \
        --video-root "$VIDEO_ROOT" --video-map "$VIDEO_MAP" --image-format png
    else
      "$py" -m vep_opsd.eipd_final_data materialize --output "$EIPD_ROOT" \
        --video-root "$VIDEO_ROOT" --image-format png
    fi
    ;;
  verify)
    "$py" -m vep_opsd.eipd_final_data verify --output "$EIPD_ROOT"
    ;;
  audit)
    "$py" -m vep_opsd.eipd_audit --data-root "$EIPD_ROOT" \
      --split train --per-stratum 50 --output "$EIPD_ROOT/audit_sheets"
    ;;
  *)
    echo "Usage: bash scripts/build_eipd_final.sh {plan|panda-subset|materialize|verify|audit}" >&2
    echo "Env: EIPD_ROOT and stage-specific EXCLUDE_*, PANDA_CSV, VIDEO_ROOT, VIDEO_MAP" >&2
    exit 2
    ;;
esac
