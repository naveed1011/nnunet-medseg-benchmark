#!/usr/bin/env bash
# =============================================================================
# STEP 1 — Experiment planning & preprocessing
#
# nnU-Net extracts the dataset fingerprint (image sizes, spacings, intensity
# statistics), then automatically plans the U-Net configurations (2d,
# 3d_fullres, and 3d_lowres/cascade for large images), and finally creates the
# preprocessed data that training will read.
#
# Usage:
#   bash scripts/01_plan_and_preprocess.sh DATASET_ID [EXTRA nnUNetv2 ARGS]
# Examples:
#   bash scripts/01_plan_and_preprocess.sh 2              # MSD Heart
#   bash scripts/01_plan_and_preprocess.sh 701            # BraTS subset
#   bash scripts/01_plan_and_preprocess.sh 701 -c 3d_fullres   # only 3d_fullres
#
# Requires env vars: nnUNet_raw, nnUNet_preprocessed, nnUNet_results
# =============================================================================
set -euo pipefail

DATASET_ID="${1:?Usage: 01_plan_and_preprocess.sh DATASET_ID [extra args]}"
shift || true

: "${nnUNet_raw:?Set nnUNet_raw env var first!}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed env var first!}"
: "${nnUNet_results:?Set nnUNet_results env var first!}"

echo "▶ Planning + preprocessing Dataset${DATASET_ID}"
echo "  raw:          ${nnUNet_raw}"
echo "  preprocessed: ${nnUNet_preprocessed}"

# --verify_dataset_integrity: catches the most common format errors (first run!)
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity "$@"

echo ""
echo "✅ Done. Inspect what nnU-Net decided:"
echo "   ${nnUNet_preprocessed}/Dataset${DATASET_ID}_*/nnUNetPlans.json     <- patch size, batch size, spacing, ..."
echo "   ${nnUNet_preprocessed}/Dataset${DATASET_ID}_*/dataset_fingerprint.json"
