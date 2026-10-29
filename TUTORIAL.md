# Run it, collect the results, push it

This is the short version. Two paths:

- **A. Local check (~15 min, CPU, no downloads)** — proves the code works before you push.
- **B. Kaggle (~1 day of GPU quota)** — produces the actual BraTS numbers and figures.

Do A first. If A fails, B will fail more expensively.

---

## A. Local check (no GPU, no dataset download)

This is the check I ran before pushing. It takes ~30 seconds.

```bash
git clone https://github.com/naveed1011/nnunet-medseg-benchmark.git
cd nnunet-medseg-benchmark

conda activate learning_env
pip install -r requirements-dev.txt
pytest -q
```

Expected: **43 passed** — or **37 passed, 6 skipped** if torch is not installed in that env
(the XAI tests skip themselves). The tests write real NIfTI files to a temp folder, evaluate
them, build a comparison table and figures, convert synthetic MSD/BraTS data, and parse every
notebook — all without downloading anything.

### Optional: the synthetic pipeline

Only if you want to watch nnU-Net itself run. **Do not do this for the 3d config on a laptop**
— I measured ~9 s/iteration, i.e. ~40 minutes for one epoch on 4 CPU cores.

```bash
pip install -r requirements.txt
python run_pipeline.py doctor      # checks torch, entry points, env vars, disk
python run_pipeline.py demo --config 2d
```

`demo` builds `Dataset900_Phantom`, plans/preprocesses, trains 1 epoch, predicts with and
without TTA, evaluates with `src/metrics.py`, writes a prediction figure, a saliency figure
and a comparison page under `demo/`.

Those numbers are a smoke test on synthetic data. They do not belong in a results table, and
they are labelled that way everywhere.

---

## B. Kaggle — the real runs

### Before anything: push the repo

The notebooks `git clone` your repo, so the first push has to happen before the GPU work.

```bash
git init -b main
git add .
git commit -m "nnU-Net vs Swin UNETR: verified pipeline, evaluation and XAI"
git remote add origin https://github.com/naveed1011/nnunet-medseg-benchmark.git
git push -u origin main
```

### Notebook settings (every notebook)

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** or **P100** |
| Internet | **On** |
| Persistence | **Save Version → Save & Run All** after every notebook |

### Step 1 — `notebooks/01_nnunet_heart_msd.ipynb` (~4–6 h)

Learning run: MSD Task02_Heart, 20 cases, `nnUNetTrainer_100epochs`, fold 0.
Download is ~2 GB from Google Drive. At the end you get:
`outputs_heart.zip` with metrics, `progress.png`, `nnUNetPlans.json`, held-out predictions.

What to look at: the printed plan (patch size, batch size, spacing) — that is nnU-Net
configuring itself, and it is the thing worth explaining in an interview.

### Step 2 — `notebooks/02_nnunet_brats.ipynb` (~8–11 h)

Attach the Kaggle dataset [`awsaf49/brats-2021-task-data`](https://www.kaggle.com/datasets/awsaf49/brats-2021-task-data)
via **Add Input**. Knobs at the top of the setup cell:

```python
MAX_CASES = 50     # 50 fits one T4 session; 100+ needs A100 or two sessions
EPOCHS    = 100    # nnUNetTrainer_{EPOCHS}epochs
```

The notebook injects a fixed 80/20 split (`brats_subset_split.json`) into
`$nnUNet_preprocessed/Dataset701_BraTS21/splits_final.json` so both arms see identical
cases. Save the version; notebook 03 attaches its output.

### Step 3 — `notebooks/03_swinunetr_brats_monai.ipynb` (~6–9 h)

Attach **notebook 02's output**. Trains MONAI Swin UNETR on exactly the same split,
validates on full volumes every 10 epochs, writes `swinunetr_best.pt` and
`results/brats_swinunetr/`.

### Step 4 — `notebooks/04_compare_nnunet_vs_swin.ipynb` (~5 min)

Attach **both** outputs. It runs `src.compare`, which writes:

```
comparison/RESULTS.md
comparison/comparison_table.{csv,md}
comparison/figures/boxplot_per_case_dice.png
comparison/figures/bar_dice_by_label.png
comparison/figures/scatter_dice_vs_hd95.png
```

Read `RESULTS.md` there. The observations are generated from the table, not by hand.

### Step 5 — `notebooks/05_xai_analysis.ipynb` (~1–2 h)

Attach notebook 03's output (and optionally 02's snapshot). Produces saliency and
GradCAM overlays for the best and worst validation cases, on 96³ crops centred on the
tumour (a full-volume backward pass does not fit on a T4 — this is stated in the figure).

### Step 6 — move results into the repo

Download the Kaggle output folders, then:

```bash
mkdir -p docs/results/brats docs/figures
cp -r <download>/comparison/*            docs/results/brats/
cp docs/results/brats/figures/*.png      docs/figures/
cp <download>/outputs_heart/...          docs/results/heart/   # optional

python -m src.compare \
  --arm nnunet=docs/results/brats/nnunet \
  --arm swinunetr=docs/results/brats/swinunetr \
  --out docs/results/brats --task "BraTS 2021 subset"
```

(Rerunning `src.compare` locally regenerates the table, figures and `RESULTS.md` from the
per-case CSVs, so the repo page and the Kaggle output cannot drift apart.)

Commit and push:

```bash
git add docs/figures docs/results
git commit -m "Add BraTS nnU-Net vs Swin UNETR results"
git push
```

### Step 7 — say what you found (the part that matters)

Open `docs/results/brats/RESULTS.md`, then write 5–10 lines in `README.md` under
*"What the numbers actually told me"*:

1. Which model won which structure, and by how much (copy the generated bullets).
2. Whether the variance changes the story — check the boxplot, not the mean.
3. What nnU-Net's plan was (patch/batch/spacing) and whether it explains the gap.
4. What you did **not** control: subset size, epoch budget, single fold, no significance test.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `not all required keys are present in dataset.json` | MSD's v1 `modality` key | run `data/convert_to_nnunet.py` (not a raw folder copy) |
| `FileNotFoundError: checkpoint_final.pth` | nnU-Net looked for `.pth`, model saved `.safetensors` | pass `-chk checkpoint_final.safetensors` |
| `CUDA out of memory` during Swin training | patch 96³ too big for your GPU | lower `ps` in `BraTSSubset` to 80, or `sw_batch_size=1` |
| Kaggle session died mid-training | 12 h limit | restore the snapshot and add `--c` |
| `No results folder named 'brats_nnunet'` | wrong output attached | attach both notebook 02 and 03 outputs |
| `plan_and_preprocess` says folder exists | stale preprocessed data | delete `$nnUNet_preprocessed/DatasetXXX_*` and rerun |

---

## What "done" looks like

- [ ] `pytest -q` passes in `learning_env` (43 tests)
- [ ] `docs/results/brats/RESULTS.md` has real numbers and a table
- [ ] `README.md` has one paragraph of your own interpretation above the table
- [ ] `docs/figures/` has the BraTS boxplot
- [ ] `.gitignore` kept the checkpoints/datasets out (`git status` shows no `.pth`, no `*_raw/`)
