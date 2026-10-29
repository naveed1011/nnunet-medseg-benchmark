#!/usr/bin/env bash
# =============================================================================
# STEP 4 — (Optional) Let nnU-Net find the best configuration & crossval setup
#
# Requires ALL 5 folds trained WITH --npz (softmax predictions saved), for the
# configurations you want it to compare (2d, 3d_fullres, ...). On a Kaggle
# budget you'll usually skip this and compare single folds instead — that's
# fine, just say so in the writeup.
#
# Usage:
#   bash scripts/04_find_best_config.sh DATASET_ID
# =============================================================================
set -euo pipefail

DATASET_ID="${1:?Usage: 04_find_best_config.sh DATASET_ID}"

nnUNetv2_find_best_configuration "${DATASET_ID}" -c 2d 3d_fullres

echo ""
echo "✅ See recommendation printed above (and in \$nnUNet_results/Dataset${DATASET_ID}_*/)."
