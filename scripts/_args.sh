#!/usr/bin/env bash
# Minimal CLI layer for the launch scripts.
#
# A job runner typically invokes a script as
#     <script path> <extra arguments>
# where there is nowhere to put a `VAR=value` prefix or an exported environment
# variable. Every knob is therefore readable from the command line:
#
#     --output-dir /path/to/run1 --nproc-per-node 8 --global-batch-size 128
#
# and the same names still work as environment variables for local runs
# (an explicit flag wins: it is parsed over the environment).
#
# Usage in a script:
#     source "$(dirname "$0")/_args.sh"
#     vepopsd_parse_args "model data output-dir" "$@"
#     # -> $MODEL, $DATA, $OUTPUT_DIR are exported;
#     #    everything unrecognised is in VEPOPSD_PASSTHROUGH (for HF/passthrough args)
#
# Machine-specific paths (MODEL, DATA, OUTPUT_ROOT, TWIFF_BENCH_DIR, ...) are read
# from <repo>/.env, which is git-ignored; .env.example documents the keys. An
# exported variable wins over .env, and a command line flag wins over both.
#
# Always accepted by every script:
#     --env-name NAME       micromamba environment name (default: vep)
#     --env-prefix PATH     explicit environment prefix
#     --config FILE         source a shell config file first; flags override it
#     --help                print the script's usage

VEPOPSD_PASSTHROUGH=()


# Set variables from a KEY=VALUE file without overriding what is already set.
# Precedence: exported environment variable > .env > script default. Command line
# flags are parsed after this and therefore win over both.
vepopsd_load_dotenv() {
    local file="$1" line key value
    [[ -f "${file}" ]] || return 0
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line#export }"
        [[ "${line}" == *=* ]] || continue
        key="${line%%=*}"
        value="${line#*=}"
        key="${key//[[:space:]]/}"
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        [[ -n "${key}" ]] || continue
        if [[ -z "${!key:-}" ]]; then
            export "${key}=${value}"
        fi
    done < "${file}"
}

VEPOPSD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VEPOPSD_REPO_ROOT
# Machine-specific paths live in .env (git-ignored); .env.example documents them.
vepopsd_load_dotenv "${VEPOPSD_REPO_ROOT}/.env"

vepopsd__list_keys() {
  local allowed="$1" key
  echo "known options (each takes a value, --key value or --key=value):"
  for key in ${allowed}; do
    echo "  --${key}"
  done
}

vepopsd_parse_args() {
  local allowed
  allowed=" env-name env-prefix config $(printf '%s' "$1") "
  shift

  VEPOPSD_PASSTHROUGH=()


# Set variables from a KEY=VALUE file without overriding what is already set.
# Precedence: exported environment variable > .env > script default. Command line
# flags are parsed after this and therefore win over both.
vepopsd_load_dotenv() {
    local file="$1" line key value
    [[ -f "${file}" ]] || return 0
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line#export }"
        [[ "${line}" == *=* ]] || continue
        key="${line%%=*}"
        value="${line#*=}"
        key="${key//[[:space:]]/}"
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        [[ -n "${key}" ]] || continue
        if [[ -z "${!key:-}" ]]; then
            export "${key}=${value}"
        fi
    done < "${file}"
}

VEPOPSD_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VEPOPSD_REPO_ROOT
# Machine-specific paths live in .env (git-ignored); .env.example documents them.
vepopsd_load_dotenv "${VEPOPSD_REPO_ROOT}/.env"

  # A platform may hand the whole scriptParam string over as a single argument;
  # re-split anything containing whitespace so both forms work.
  local -a args=()
  local token
  for token in "$@"; do
    if [[ "${token}" == *[[:space:]]* ]]; then
      # shellcheck disable=SC2206
      args+=( ${token} )
    else
      args+=( "${token}" )
    fi
  done
  set -- ${args[@]+"${args[@]}"}

  # --config is resolved first, so explicit flags always win over it.
  local -a configs=()
  local -a without_configs=()
  while (( $# )); do
    case "$1" in
      --config=*) configs+=("${1#*=}"); shift ;;
      --config)
        [[ -n "${2:-}" ]] || { echo "--config needs a file" >&2; exit 2; }
        configs+=("$2"); shift 2 ;;
      *) without_configs+=("$1"); shift ;;
    esac
  done
  set -- ${without_configs[@]+"${without_configs[@]}"}
  for token in ${configs[@]+"${configs[@]}"}; do
    [[ -s "${token}" ]] || { echo "config file not found or empty: ${token}" >&2; exit 2; }
    # shellcheck disable=SC1090
    source "${token}"
  done

  local key value name var
  while (( $# )); do
    token="$1"
    case "${token}" in
      --help|-h)
        if declare -F vepopsd_usage >/dev/null; then
          vepopsd_usage
        else
          vepopsd__list_keys "${allowed}"
        fi
        exit 0 ;;
      --)
        shift
        VEPOPSD_PASSTHROUGH+=("$@")
        break ;;
      --*=*)
        key="${token%%=*}"; value="${token#*=}"; shift ;;
      --*)
        key="${token}"; value=""; shift ;;
      *)
        VEPOPSD_PASSTHROUGH+=("${token}"); shift; continue ;;
    esac

    name="${key#--}"
    if [[ "${allowed}" != *" ${name} "* ]]; then
      # Unknown flag: hand it to the underlying entry point unchanged. Its value
      # (if any) is a bare token and flows through the branch above.
      VEPOPSD_PASSTHROUGH+=("${key}")
      continue
    fi
    if [[ -z "${value}" ]]; then
      if [[ -z "${1:-}" || "${1}" == --* ]]; then
        echo "${key} needs a value (--${name} VALUE)" >&2
        exit 2
      fi
      value="$1"; shift
    fi
    var="${name^^}"
    var="${var//-/_}"
    # Universal keys keep the prefixed names that scripts/_env.sh reads.
    case "${name}" in
      env-name) var="VEP_OPSD_ENV_NAME" ;;
      env-prefix) var="VEP_OPSD_ENV_PREFIX" ;;
    esac
    export "${var}=${value}"
  done
}
