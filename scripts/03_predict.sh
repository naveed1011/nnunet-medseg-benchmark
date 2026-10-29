#!/usr/bin/env bash
# =============================================================================
# STEP 3 — Inference (predictions)
#
# Runs a trained model over a folder of test images and writes .nii.gz
# predictions. Use -f all for a 5-fold ensemble (best accuracy, needs all
# folds trained), or a single fold number when training budget was limited.
#
# Usage:
#   bash scripts/03_predict.sh DATASET_ID [CONFIG] [FOLD_OR_all] [INPUT_DIR] [OUTPUT_DIR]
# Examples:
#   bash scripts/03_predict.sh 2 3d_fullres 0
#   bash scripts/03_predict.sh 2 3d_fullres all "$nnUNet_raw/Dataset002_Heart/imagesTr" preds/heart
# =============================================================================
set -euo pipefail

DATASET_ID="${1:?Usage: 03_predict.sh DATASET_ID [CONFIG] [FOLD|all] [INPUT_DIR] [OUTPUT_DIR]}"
CONFIG="${2:-3d_fullres}"
FOLD="${3:-0}"

INPUT_DIR="${4:-${nnUNet_raw}/Dataset$(printf '%03d' "${DATASET_ID}")_*/imagesTr}"
# shellcheck disable=SC2086
INPUT_DIR=$(echo ${INPUT_DIR})  # resolve simple globs
OUTPUT_DIR="${5:-predictions/Dataset${DATASET_ID}_${CONFIG}_fold${FOLD}}"

mkdir -p "${OUTPUT_DIR}"

echo "▶ Predicting: Dataset${DATASET_ID} ${CONFIG} folds=${FOLD}"
echo "  input:  ${INPUT_DIR}"
echo "  output: ${OUTPUT_DIR}"

# --save_probabilities enables ensembling later (not needed for a single model)
nnUNetv2_predict \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -d "${DATASET_ID}" \
  -c "${CONFIG}" \
  -f ${FOLD} \
  -chk checkpoint_final.pth \
  2>/dev/null || \
nnUNetv2_predict \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -d "${DATASET_ID}" \
  -c "${CONFIG}" \
  -f ${FOLD} \
  -chk checkpoint_final.safetensors

echo ""
echo "✅ Predictions written to ${OUTPUT_DIR}"
echo "   Evaluate them with:  python -m src.metrics --pred ${OUTPUT_DIR} --gt <labelsTr dir>"
