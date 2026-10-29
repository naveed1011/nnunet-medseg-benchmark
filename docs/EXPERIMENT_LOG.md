# 🧪 Experiment log

> The honest lab notebook of this project. Every run gets an entry — including the ones that
> failed. I fill these in as I go, not from memory.

Template per run (copy-paste):

```
### RUN-XXX — <one-line description>
- Date / hardware:             e.g. 2026-11-02, Kaggle T4, session 1 of 1
- Dataset / subset / split:    e.g. BraTS2021, first 50 cases, 40/10 fixed
- Config:                      e.g. nnUNetTrainer_100epochs, 3d_fullres, fold 0, --npz
- Planned patch/batch:         from nnUNetPlans.json
- Wall-clock:                  preprocess / train / predict / evaluate
- Results:                     Dice + HD95 per label, and the honest caveat
- Observations:                what the curves/figures show; surprises
- Verdict:                     ✅ keep · 🔁 rerun with change · ❌ dead end
- Artifacts:                   output link / local path / commit hash
```

---

## Log

### RUN-001 — Code verification on a laptop; training deliberately stopped
- Date / hardware: 2026-10-29, Windows 10, 4 CPU cores, torch CPU-only (no CUDA)
- Dataset / split: synthetic `Dataset900_Phantom`, 12 cases, 9 train / 3 val, fixed
- What ran: `pytest` (43 pass) · dataset integrity · fingerprint → plan → preprocess (147 s) ·
  `nnUNetv2_train` launch for `3d_fullres` and `2d`
- Planner output: `3d_fullres` patch `[56, 80, 64]`, batch 2, spacing `[1.5, 1.0, 1.0]`;
  `2d` patch `[80, 64]`, batch 24
- Measured cost: `3d_fullres` ≈ 9 s/iteration → ≈ 40 min for one 250-iteration epoch; the
  `2d` run had not finished epoch 0 after 11 min on 4 cores
- Decision: stop training here and run it on Kaggle; this is a hardware limit, not a bug
- Results: no Dice reported — one epoch on synthetic data would not mean anything
- Verdict: ✅ code verified · ⏸ training deferred to GPU
- Artifacts: `docs/figures/demo_data_qc.png`, `tests/`, planner output in this entry

### RUN-002 — MSD Task02_Heart, 100-epoch nnU-Net (queued)
- Dataset / split: 20 cases, nnU-Net default fold 0 (16/4)
- Config: `nnUNetTrainer_100epochs`, `3d_fullres`, fold 0, `--npz`
- Expected: preprocess ~5 min, train ~4 h on T4, predict ~5 min
- Results: _to fill from `notebooks/01_nnunet_heart_msd.ipynb`_
- Verdict: ⏳ queued

### RUN-003 — BraTS 2021 subset, both arms (queued)
- Dataset / split: first 50 cases, 40/10 fixed (`brats_subset_split.json`)
- nnU-Net: `nnUNetTrainer_100epochs`, `3d_fullres`, fold 0
- Swin UNETR: 96³ crops, batch 1, 100 epochs, full-volume validation every 10
- Results: _to fill from notebooks 02–04; `src.compare` writes the table_
- Verdict: ⏳ queued

---

## Things that broke (and what fixed them)

| What happened | Cause | Fix |
|---|---|---|
| Phantom tumour was off-centre in the first QC figure | coordinate arrays paired with the wrong spacing (z vs x) | rebuilt the grid per axis in `scripts/make_demo_data.py` |
| `plan_and_preprocess` would have rejected a copied MSD folder | MSD `dataset.json` uses the v1 `modality` key; v2 requires `channel_names` | notebook 01 now runs `data/convert_to_nnunet.py` instead of copying |
| `verify_dataset_integrity` crashed on the converter's output | I had written a `"dataset"` block with `"image"` instead of `"images"` | removed the block; nnU-Net discovers files from `imagesTr/` |
| Swin validation Dice was measured on random 96³ crops | `val_ds[i]` returns a random crop | added `validate_full()` — sliding window over the whole volume |
| `pick("nnunet")` matched `brats_swinunetr` | substring matching | exact parent-folder-name match |
| BraTS labels silently rescaled on write | reused the source NIfTI header (scl_slope) | fresh header + uint8 dtype in the converter |

---

## Backlog

- [ ] 250-epoch runs for both arms if quota allows (closer to converged)
- [ ] `resenc` plans preset vs default plans (nnU-Net v2 recommendation)
- [ ] Post-processing on/off ablation (`nnUNetv2_apply_postprocessing`)
- [ ] 5-fold instead of 1-fold for a variance estimate
- [ ] Phase 5: MAE-pretrained encoder inside a custom `nnUNetTrainer` subclass
- [ ] Match Swin UNETR and nnU-Net on *iterations*, not epochs (needs a second Kaggle session)
