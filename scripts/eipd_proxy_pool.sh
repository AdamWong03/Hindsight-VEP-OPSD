#!/usr/bin/env bash
# Run the restartable EIPD clip downloader over every healthy local proxy.
#
# The pool performs a real yt-dlp probe (not merely a TCP/curl check), removes
# duplicate public egress IPs, assigns one deterministic download shard to each
# healthy egress, and keeps retrying a lane when its proxy disappears.  Every
# event and downloader record is written to one timestamped aggregate log.
#
# Required/typical environment:
#   EIPD_ROOT=/data1/wyd/datasets/EIPD-Data-v1
#   PYTHON=/data1/envs/opsd/bin/python
#   REUSE_VIDEO_ROOT=/data1/wyd/datasets/TwiFF-Privileged-50K/videos
#
# Optional knobs:
#   PROXY_PORTS="6088 6101 6102"
#   POOL_WORKERS_PER_PROXY=2
#   POOL_PROBE_INTERVAL=60
#   POOL_STATUS_INTERVAL=300
#   POOL_AUTH_BREAKER=50
#   POOL_PROBE_VIDEO_ID=nXJrBDHV5hA
#   POOL_DISCOVER_ONLY=1
#   POOL_LOG=/path/to/aggregate.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EIPD_ROOT="${EIPD_ROOT:-/data1/wyd/datasets/EIPD-Data-v1}"
PYTHON="${PYTHON:-/data1/envs/opsd/bin/python}"
VIDEO_ROOT="${VIDEO_ROOT:-${EIPD_ROOT}/raw_clips}"
REUSE_VIDEO_ROOT="${REUSE_VIDEO_ROOT:-/data1/wyd/datasets/TwiFF-Privileged-50K/videos}"
# Only use the project's explicitly authorised VPN endpoints by default.
# Additional ports must be opted into explicitly through PROXY_PORTS.
PROXY_PORTS="${PROXY_PORTS:-6088 6101 6102}"
POOL_WORKERS_PER_PROXY="${POOL_WORKERS_PER_PROXY:-2}"
POOL_PROBE_INTERVAL="${POOL_PROBE_INTERVAL:-60}"
POOL_STATUS_INTERVAL="${POOL_STATUS_INTERVAL:-300}"
POOL_AUTH_BREAKER="${POOL_AUTH_BREAKER:-50}"
POOL_PROBE_VIDEO_ID="${POOL_PROBE_VIDEO_ID:-nXJrBDHV5hA}"
POOL_DISCOVER_ONLY="${POOL_DISCOVER_ONLY:-0}"
POOL_LOG="${POOL_LOG:-${EIPD_ROOT}/logs/proxy-pool.log}"

case "${1:-}" in
  -h|--help)
    sed -n '2,/^set -uo pipefail/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  "") ;;
  *) echo "unknown argument: $1 (configuration uses environment variables)" >&2; exit 2 ;;
esac

[[ -x "${PYTHON}" ]] || { echo "python is not executable: ${PYTHON}" >&2; exit 2; }
[[ -d "${EIPD_ROOT}" ]] || { echo "EIPD root does not exist: ${EIPD_ROOT}" >&2; exit 2; }
mkdir -p "$(dirname "${POOL_LOG}")" "${VIDEO_ROOT}"
touch "${POOL_LOG}"

pool_log() {
  # One append per record keeps concurrent lane messages intact on local filesystems.
  printf '%s %s\n' "$(date -Is)" "$*" >> "${POOL_LOG}"
  printf '%s %s\n' "$(date -Is)" "$*"
}

proxy_ip() {
  local port="$1"
  curl --proxy "http://127.0.0.1:${port}" --connect-timeout 4 --max-time 8 \
    --silent --show-error https://api.ipify.org 2>/dev/null
}

probe_proxy() {
  local port="$1"
  timeout 30s "${PYTHON}" -m yt_dlp \
    --proxy "http://127.0.0.1:${port}" \
    --no-warnings --skip-download --print id \
    "https://www.youtube.com/watch?v=${POOL_PROBE_VIDEO_ID}" \
    >/dev/null 2>&1
}

declare -a ACTIVE_PORTS=()
declare -a ACTIVE_IPS=()
declare -A SEEN_IPS=()

pool_log "event=discover_start candidates=\"${PROXY_PORTS}\""
for port in ${PROXY_PORTS}; do
  ip="$(proxy_ip "${port}" || true)"
  if [[ -z "${ip}" ]]; then
    pool_log "event=probe port=${port} status=DOWN reason=no_egress_ip"
    continue
  fi
  if ! probe_proxy "${port}"; then
    pool_log "event=probe port=${port} ip=${ip} status=BLOCKED reason=yt_dlp_probe_failed"
    continue
  fi
  if [[ -n "${SEEN_IPS[${ip}]:-}" ]]; then
    pool_log "event=probe port=${port} ip=${ip} status=DUPLICATE kept_port=${SEEN_IPS[${ip}]}"
    continue
  fi
  SEEN_IPS["${ip}"]="${port}"
  ACTIVE_PORTS+=("${port}")
  ACTIVE_IPS+=("${ip}")
  pool_log "event=probe port=${port} ip=${ip} status=READY"
done

SHARD_COUNT="${#ACTIVE_PORTS[@]}"
if (( SHARD_COUNT == 0 )); then
  pool_log "event=pool_exit status=NO_HEALTHY_PROXY"
  exit 1
fi
pool_log "event=pool_ready lanes=${SHARD_COUNT} workers_per_lane=${POOL_WORKERS_PER_PROXY}"
if [[ "${POOL_DISCOVER_ONLY}" == "1" ]]; then
  exit 0
fi

declare -a CHILD_PIDS=()
STOPPING=0

cleanup() {
  (( STOPPING )) && return
  STOPPING=1
  pool_log "event=pool_stop children=${#CHILD_PIDS[@]}"
  local pid
  for pid in "${CHILD_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

download_lane() {
  local shard="$1" port="$2" ip="$3" rc line
  local label="lane=${shard}/${SHARD_COUNT} port=${port} ip=${ip}"
  while true; do
    if ! probe_proxy "${port}"; then
      pool_log "${label} event=proxy_wait status=DOWN retry_seconds=${POOL_PROBE_INTERVAL}"
      sleep "${POOL_PROBE_INTERVAL}"
      continue
    fi
    pool_log "${label} event=download_start status=UP"
    set +o pipefail
    stdbuf -oL -eL env PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON}" -u -m vep_opsd.eipd_clips \
      --output "${EIPD_ROOT}" \
      --clip-root "${VIDEO_ROOT}" \
      --workers "${POOL_WORKERS_PER_PROXY}" \
      --reserve-factor 1.25 \
      --reuse-root "${REUSE_VIDEO_ROOT}" \
      --proxy "http://127.0.0.1:${port}" \
      --shard-count "${SHARD_COUNT}" \
      --shard-index "${shard}" \
      --min-sleep 0.20 --max-sleep 0.60 --sleep-requests 0.20 \
      --auth-circuit-breaker "${POOL_AUTH_BREAKER}" 2>&1 |
      while IFS= read -r line; do
        pool_log "${label} download=${line}"
      done
    rc="${PIPESTATUS[0]}"
    set -o pipefail
    pool_log "${label} event=download_exit rc=${rc} retry_seconds=${POOL_PROBE_INTERVAL}"
    if [[ "${rc}" == "0" ]]; then
      break
    fi
    sleep "${POOL_PROBE_INTERVAL}"
  done
}

status_reporter() {
  local summary
  while true; do
    summary="$("${PYTHON}" - "${EIPD_ROOT}" <<'PY'
import collections
import json
import sqlite3
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {}
state = root / "download_state.sqlite"
if state.exists():
    con = sqlite3.connect(state, timeout=30)
    payload["download_state"] = dict(collections.Counter(
        row[0] for row in con.execute("SELECT status FROM downloads")
    ))
    con.close()
report = root / "materialize_report.json"
if report.exists():
    accepted = json.loads(report.read_text()).get("accepted_this_pass", {})
    targets = {
        "train/Predictive": 30000, "train/Instructional": 15000,
        "train/Camera": 5000, "val/Predictive": 3000,
        "val/Instructional": 1500, "val/Camera": 500,
    }
    payload["accepted"] = accepted
    payload["missing"] = {
        key: max(0, target - int(accepted.get(key, 0)))
        for key, target in targets.items()
    }
print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
PY
)"
    pool_log "event=summary state=${summary}"
    sleep "${POOL_STATUS_INTERVAL}"
  done
}

status_reporter &
CHILD_PIDS+=("$!")
for (( shard=0; shard<SHARD_COUNT; shard++ )); do
  download_lane "${shard}" "${ACTIVE_PORTS[${shard}]}" "${ACTIVE_IPS[${shard}]}" &
  CHILD_PIDS+=("$!")
done

pool_log "event=pool_started lane_pids=\"${CHILD_PIDS[*]}\""
wait
