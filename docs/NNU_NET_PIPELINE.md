# 🔬 The nnU-Net v2 pipeline — what actually happens at each step

> A conceptual companion to the notebooks. The goal: understand nnU-Net deeply
> enough to explain it in an interview, not just run it.

nnU-Net ("no-new-Net") is a **self-configuring** segmentation framework
(Isensee et al., *Nature Methods* 2021). Its thesis: most "novel architecture"
gains disappear once you adapt pipeline details (preprocessing, patch size,
augmentation, training length) properly to each dataset. nnU-Net automates
that adaptation — and usually wins or ties SOTA as a result.

```
raw dataset ──▶ fingerprint ──▶ plans ──▶ preprocessed data ──▶ train ──▶ predict
   (you)        (automatic)   (automatic)     (automatic)      (5-fold)  (sliding window)
```

---

## Step 0 — Dataset format (`DatasetXXX_Name`)

```
nnUNet_raw/Dataset002_Heart/
├── dataset.json        # channels, labels, numTraining, file_ending
├── imagesTr/           # heart_01_0000.nii.gz  <- _0000 = channel 0
└── labelsTr/           # heart_01.nii.gz       <- integer labels, NO suffix
```

Rules that trip everyone up:
- Dataset IDs are 3 digits; folder must be `DatasetXXX_Name` (v1 used `TaskXXX`).
- Channel suffix `_0000`, `_0001`, … is **mandatory**, even for single-channel data.
- Labels must be **consecutive integers starting at 0** (BraTS `4`→`3` remap!).
- All files in one dataset must share the same extension (`.nii.gz` here).
- `dataset.json` declares `channel_names` and `labels` — v2 validates this hard.

## Step 1 — Fingerprint extraction

nnU-Net samples the training cases and measures: image shapes, voxel spacings,
intensity distributions (percentiles per modality), class sizes, and how much
memory the data would need. Output: `dataset_fingerprint.json`.

**Why it matters:** every later decision is derived from these numbers. When a
model underperforms, checking the fingerprint (e.g. wildly anisotropic spacing,
tiny foreground classes) is the first diagnostic step.

## Step 2 — Experiment planning (`nnUNetPlans.json`)

The planner turns the fingerprint + your GPU memory assumptions into concrete
configurations:

| Decision | Example (Heart) | Example (BraTS-like) |
|---|---|---|
| target spacing | median spacing, downsampling capped | 1×1×1 mm (already resampled data) |
| patch size | e.g. `[48, 224, 224]` (anisotropic!) | e.g. `[128, 128, 128]` |
| batch size | 2 | 2 |
| architecture depth | #stages from image size | deeper for larger patches |
| normalization | per channel: z-score if CT-like, else robust rescale | `CT`/`MR` schemes per modality |
| configurations | `2d`, `3d_fullres` (+`3d_lowres`/cascade if images are huge) | same |

Key insight to be able to articulate: **patch size is chosen to (a) fit GPU
memory with the planned batch size and (b) cover enough context** — the "receptive
field vs. structure size" trade-off is automated.

## Step 3 — Preprocessing

For each configuration, images are resampled to the target spacing (3rd-order
spline for data, nearest-neighbor-ish for labels), normalized, and stored as
`.npy`/`.npz` + per-case `.pkl` property files. Training then reads these —
which is why training epochs are fast and why you must **re-run preprocessing
if you change the plan**.

## Step 4 — Training

- Default trainer: `nnUNetTrainer` — 1000 epochs × 250 iterations, SGD with
  poly LR, heavy augmentation (rotation, scaling, noise, blur, brightness,
  gamma, mirror), deep supervision, Dice+CE loss.
- Reduced presets: `nnUNetTrainer_50/100/250/500epochs` (used here for Kaggle budgets).
- 5-fold CV by default; splits live in `splits_final.json` in the preprocessed
  folder — **you can overwrite them** (this repo injects a fixed split so the
  Swin UNETR arm trains on identical data).
- Outputs per fold: `checkpoint_final.*` (weights), `progress.png` (curves),
  `validation/` (predictions + `summary.json` with per-case metrics).
- `--npz` saves validation softmax outputs → needed for ensembling /
  `nnUNetv2_find_best_configuration`.
- `--c` continues an interrupted run from `checkpoint_latest`.

## Step 5 — Inference & evaluation

- `nnUNetv2_predict` runs **sliding-window** inference: the volume is covered
  with overlapping patches (Gaussian-weighted), predictions stitched, then
  resampled back to original geometry.
- Test-time augmentation = mirroring along axes (on by default in predict).
- `-f all` ensembles the 5 fold models (the number nnU-Net papers report).
- `nnUNetv2_apply_postprocessing` can enforce connectivity rules learned from CV.

## Reading `plans.json` like a pro (interview fuel)

```json
"3d_fullres": {
  "patch_size": [48, 224, 224],     // anisotropic patch for anisotropic spacing
  "batch_size": 2,
  "spacing": [0.86, 0.86, 0.86],    // target spacing after resampling
  "normalization_schemes": {"0": "CT"},  // per-channel normalization
  "n_conv_per_stage_encoder": [2,2,2,2,2],  // architecture depth/width
  "resampling_fn_data": "resample_data_order_3"
}
```

## Common failure modes & fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Dataset integrity check failed` | bad suffixes / label values / json mismatch | follow Step 0 rules exactly |
| Training instantly OOM | plans assume more VRAM than you have | re-plan with less: `nnUNetv2_plan_and_preprocess -pl ResamplingPlanner ...` or reduce `gpu_memory_target_in_GB` in custom plans |
| Dice ≈ 0 on small structures | class imbalance / too few epochs | more epochs, check fingerprint class sizes |
| Predictions misaligned | affine/spacing inconsistency in raw data | re-check NIfTI headers |
| Kaggle session died | 12 h limit | snapshot + `--c` resume (see KAGGLE_GUIDE) |

## What nnU-Net v2 (2024+) adds

- **Residual-encoder U-Net presets** (`-c 3d_fullres` with `resenc` plans) — the
  new recommended defaults, stronger on most tasks (Isensee et al., arXiv 2404.08244).
- Checkpoints moving to `.safetensors` (this repo handles both).
- Cleaner trainer/trainer-variant system — you can subclass `nnUNetTrainer`
  to swap losses/augmentations without touching the pipeline (useful for the
  planned **Phase 5: MAE-pretrained encoder** experiment).

## References

- Isensee et al., *nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation*, Nature Methods 2021.
- Isensee et al., *Extending nnU-Net is all you need*, arXiv 2024.
- Official docs: https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation
