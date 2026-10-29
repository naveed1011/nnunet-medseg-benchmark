#!/usr/bin/env bash
# =============================================================================
# STEP 2 — Training
#
# FULL training (1000 epochs, 5 folds, ~2 days per configuration on a modern
# GPU) is usually not feasible on a free Kaggle 12h session. This script
# supports both modes:
#
#   full:    bash scripts/02_train.sh DATASET_ID CONFIG FOLD
#            e.g. bash scripts/02_train.sh 2 3d_fullres 0
#
#   kaggle:  KAGGLE_MODE=1 bash scripts/02_train.sh DATASET_ID CONFIG FOLD [EPOCHS]
#            Uses nnUNetTrainer_100epochs (a built-in reduced-epoch trainer)
#            so a run fits inside one Kaggle GPU session. Perfect for the
#            pipeline demo + a meaningful (if not SOTA) comparison.
#
# Resume an interrupted training with:  nnUNetv2_train ... --c
# =============================================================================
set -euo pipefail

DATASET_ID="${1:?Usage: 02_train.sh DATASET_ID CONFIG FOLD [EPOCHS]}"
CONFIG="${2:-3d_fullres}"
FOLD="${3:-0}"
EPOCHS="${4:-100}"

if [[ "${KAGGLE_MODE:-0}" == "1" ]]; then
  TRAINER="nnUNetTrainer_${EPOCHS}epochs"
  echo "▶ Kaggle mode: training Dataset${DATASET_ID} ${CONFIG} fold ${FOLD}"
  echo "  trainer=${TRAINER} (reduced epochs to fit a 12h session)"
  echo "  tip: checkpoints land in \$nnUNet_results — copy them to /kaggle/working"
  echo "       before the session ends, then attach that output as a Dataset later."
  nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" -tr "${TRAINER}" --npz -device cuda
else
  echo "▶ Full training: Dataset${DATASET_ID} ${CONFIG} fold ${FOLD} (1000 epochs)"
  nnUNetv2_train "${DATASET_ID}" "${CONFIG}" "${FOLD}" --npz
fi

echo ""
echo "✅ Training finished. Outputs in:"
echo "   \$nnUNet_results/Dataset${DATASET_ID}_*/nnUNetTrainer*__nnUNetPlans__${CONFIG}/fold_${FOLD}/"
echo "   - progress.png            (loss / pseudo-dice curves)"
echo "   - checkpoint_final.*      (used for validation & inference)"
echo "   - validation/summary.json (per-case + mean validation metrics)"
