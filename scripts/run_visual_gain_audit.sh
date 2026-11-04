#!/usr/bin/env bash
# Run only after the two pilots; supervised, resumable full 5K evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN=/data2/wyd/experiments/vep_visual_gain_audit_ckpt400_20261105
DATA=/data2/wyd/datasets/EIPD-Data-v1-visual-gain-val5k
PY=/data1/wyd/envs/mlip/bin/python
export PYTHONPATH="$ROOT"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$RUN/full" "$RUN/report"
cd "$ROOT"
echo "[$(date -Is)] verifying completed category-stratified pilots"
"$PY" -m vep_opsd.visual_gain_report --results "$RUN/pilot" --data-root "$DATA" \
  --output "$RUN/pilot-report" --bootstrap-repeats 200 --pilot
git rev-parse HEAD > "$RUN/code_commit.txt"
git diff > "$RUN/code_uncommitted.patch"
cp "$DATA/data-config.json" "$RUN/data-config.json"
cp "$DATA/matching.json" "$RUN/matching.json"
echo "[$(date -Is)] capturing immutable local weight/config hashes once"
/usr/bin/python3 "$ROOT/scripts/capture_visual_gain_manifest.py" --output "$RUN/resource-manifest.json"

launch() {
  local family="$1" shard="$2" gpu="$3"
  local unit="visual-gain-$family-$shard"
  if [[ -f "$RUN/full/$family-$shard.complete" ]]; then return; fi
  if systemctl --user is-active --quiet "$unit"; then return; fi
  systemd-run --user --unit="$unit" --description="Full 5K visual-gain $family shard $shard/2" \
    --property="WorkingDirectory=$ROOT" --property=MemoryMax=40G --property=MemorySwapMax=0 \
    --property=Restart=on-failure --property=RestartSec=60 \
    --property=StartLimitIntervalSec=3600 --property=StartLimitBurst=3 \
    --property="StandardOutput=append:$RUN/$family-$shard.log" --property=StandardError=inherit \
    /usr/bin/bash "$ROOT/scripts/run_visual_gain_worker.sh" "$family" "$shard" 2 "$gpu" full
}

# The old trainer has been stopped. Two 8B inference copies fit on PRO 0;
# other workers share PRO 3 and L40S 5 only after the free-memory gate passes.
launch base 0 GPU-f48b2c84-f983-64e3-000b-d669b9e64454
launch cdlsd 0 GPU-ed405bf3-4069-2d20-0432-9ecda26de366
launch base 1 GPU-9e62da9a-1b28-25c8-e1a0-e96004a11a95
launch cdlsd 1 GPU-f48b2c84-f983-64e3-000b-d669b9e64454

while true; do
  complete=0
  for family in base cdlsd; do
    for shard in 0 1; do
      if [[ -f "$RUN/full/$family-$shard.complete" ]]; then
        complete=$((complete + 1))
      elif systemctl --user is-failed --quiet "visual-gain-$family-$shard"; then
        echo "[$(date -Is)] failed: visual-gain-$family-$shard; inspect $RUN/$family-$shard.log"
        exit 1
      fi
    done
  done
  records=0
  for file in "$RUN"/full/*.jsonl; do
    if [[ -f "$file" ]]; then records=$((records + $(wc -l < "$file"))); fi
  done
  echo "[$(date -Is)] complete_workers=$complete/4 result_records=$records/20090"
  if (( complete == 4 )); then break; fi
  sleep 30
done
echo "[$(date -Is)] generating category statistics, source-bootstrap intervals and semantic review pack"
"$PY" -m vep_opsd.visual_gain_report --results "$RUN/full" --data-root "$DATA" \
  --output "$RUN/report" --bootstrap-repeats 2000
echo "[$(date -Is)] FULL NUMERIC AUDIT COMPLETE: $RUN/report/report.md; independent semantic annotation pending"
