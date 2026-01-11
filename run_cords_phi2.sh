#!/bin/bash
# ================================================================
# Script: run_cords_phi2.sh
#
# Purpose:
#   Run cords_phi2 with
#   dataset-specific sample configurations.
# ================================================================

set -euo pipefail

# ========================
# 可执行程序
# ========================
BIN="./cords_phi2"

# ========================
# 数据集配置
# ========================
declare -A DATASETS=(
  ["dmv"]="./dataset_public/DMV/dmv_withhead_11cols.csv"
  ["census"]="./dataset_public/Census/census.csv"
  ["instacart"]="./dataset_public/Instacart/order_wide_clean.csv"
)

# ========================
# 每个数据集对应的 sample 列表
# （这是你关心的核心）
# ========================
declare -A DATASET_SAMPLES=(
  ["dmv"]="1000 2000 4000 8000 16000 48842"
  ["census"]="1000 2000 4000 8000 16000 11591877"
  ["instacart"]="1000 2000 4000 8000 16000 33819106"
)

# ========================
# 输出与日志根目录
# ========================
RESULT_ROOT="./cords_results"
LOG_ROOT="./cords_logs"

# ========================
# 主循环
# ========================
for DATASET in "${!DATASETS[@]}"; do
    CSV_FILE="${DATASETS[$DATASET]}"
    SAMPLE_LIST="${DATASET_SAMPLES[$DATASET]:-}"

    if [[ ! -f "${CSV_FILE}" ]]; then
        echo "[WARN] CSV not found: ${CSV_FILE}, skip ${DATASET}"
        continue
    fi

    if [[ -z "${SAMPLE_LIST}" ]]; then
        echo "[WARN] No samples defined for ${DATASET}, skip"
        continue
    fi

    echo "================================================"
    echo "[INFO] Dataset: ${DATASET}"
    echo "[INFO] Samples: ${SAMPLE_LIST}"
    echo "================================================"

    OUT_DIR="${RESULT_ROOT}/${DATASET}/phi2"
    LOG_DIR="${LOG_ROOT}/${DATASET}/phi2"

    mkdir -p "${OUT_DIR}"
    mkdir -p "${LOG_DIR}"

    for SAMPLE in ${SAMPLE_LIST}; do
        echo "[INFO]   Sample size: ${SAMPLE}"

        OUT_FILE="${OUT_DIR}/phi2_${DATASET}_sample${SAMPLE}.txt"
        LOG_FILE="${LOG_DIR}/phi2_${DATASET}_sample${SAMPLE}.log"

        /usr/bin/time -v \
            ${BIN} "${CSV_FILE}" \
            --sample "${SAMPLE}" \
            --out "${OUT_FILE}" \
            > "${LOG_FILE}" 2>&1
    done
done

echo "[DONE] All dataset-specific experiments finished."
