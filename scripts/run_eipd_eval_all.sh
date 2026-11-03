#!/usr/bin/env bash
# 临时脚本：两组 checkpoint × 三种 latent 干预，全部 8 卡分片评测，最后打印对照表。
#
#   bash scripts/run_eipd_eval_all.sh                 # 6 组全跑
#   ABLATIONS=none bash scripts/run_eipd_eval_all.sh  # 只跑基线（约 40 分钟）
#   FORCE=1 bash scripts/run_eipd_eval_all.sh         # 已有的结果也重跑
#
# 结果落在各自 checkpoint 目录下的 eval.<ablate>-k<K>.json，跳过已完成的那几组。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/_env.sh
source "${SCRIPT_DIR}/_env.sh"
PY="${VEP_OPSD_ENV_PREFIX}/bin/python"

OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/ofs-llm-ssd/user/shengrenren_i/outputs}"
K16_DIR="${K16_DIR:-${OUTPUT_ROOT}/eipd/v2-full-20261102-110324/stage1}"
K4_DIR="${K4_DIR:-${OUTPUT_ROOT}/eipd/v2-short-gain-20261103-120458/stage1}"
ABLATIONS="${ABLATIONS:-none zero shuffle}"
FORCE="${FORCE:-0}"

mkdir -p "${ROOT}/logs"

run_one() {
    local directory="$1" steps="$2" ablate="$3"
    local tag="${ablate}-k${steps}"
    local merged="${directory}/eval.${tag}.json"

    if [[ -s "${merged}" && "${FORCE}" != "1" ]]; then
        echo "  跳过 ${tag}（已有 ${merged}）"
        return 0
    fi
    echo "  评测 ${tag}  <- ${directory}"
    bash "${SCRIPT_DIR}/eval_eipd_multi.sh" \
        --checkpoint "${directory}" --latent-steps "${steps}" --ablate "${ablate}"
}

failed=0
for ablate in ${ABLATIONS}; do
    echo
    echo "===== ablate=${ablate} ====="
    run_one "${K16_DIR}" 16 "${ablate}" || failed=$((failed + 1))
    run_one "${K4_DIR}" 4 "${ablate}" || failed=$((failed + 1))
done

echo
echo "===== 汇总 ====="
export K16_DIR K4_DIR ABLATIONS
"${PY}" - <<'PY'
import json, os
from pathlib import Path

runs = [("K=16", os.environ["K16_DIR"], 16), ("K=4", os.environ["K4_DIR"], 4)]
rows = []
for name, directory, steps in runs:
    for ablate in os.environ["ABLATIONS"].split():
        path = Path(directory) / f"eval.{ablate}-k{steps}.json"
        if not path.is_file():
            rows.append((name, ablate, None, None, 0))
            continue
        summary = json.loads(path.read_text())["summary"]
        rows.append((name, ablate, summary["exact_match"], summary["token_f1"],
                     summary["samples"]))

base = {name: (exact, f1) for name, ablate, exact, f1, _ in rows if ablate == "none"}

print(f"{'arm':6} {'ablate':8} {'exact':>8} {'Δ vs none':>10} {'token_f1':>9} {'Δ':>8} {'n':>6}")
print("-" * 60)
for name, ablate, exact, f1, samples in rows:
    if exact is None:
        print(f"{name:6} {ablate:8} {'—':>8} {'—':>10} {'—':>9} {'—':>8} {0:>6}")
        continue
    be, bf = base.get(name, (None, None))
    de = f"{exact - be:+.4f}" if be is not None and ablate != "none" else ""
    df = f"{f1 - bf:+.4f}" if bf is not None and ablate != "none" else ""
    print(f"{name:6} {ablate:8} {exact:8.4f} {de:>10} {f1:9.4f} {df:>8} {samples:>6}")

print()
print("读法：zero/shuffle 的 Δ 越负，说明答案越依赖 latent；接近 0 说明模型不看 latent 也能答。")
PY

if (( failed > 0 )); then
    echo
    echo "${failed} 组评测失败，见 logs/eval-*.log" >&2
    exit 1
fi
