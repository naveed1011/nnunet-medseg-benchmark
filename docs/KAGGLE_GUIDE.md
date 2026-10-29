# ☁️ Kaggle Guide — running this project on free GPUs

Everything in this repo is designed to run inside Kaggle's free GPU quota
(T4 x2 or P100, ~30 h/week with phone verification). This guide covers the
platform quirks that bite people the first time.

---

## 1. Notebook settings checklist

| Setting | Value | Why |
|---|---|---|
| Accelerator | **GPU T4 x2** (or P100) | nnU-Net/Swin UNETR need CUDA |
| Internet | **On** | `pip install nnunetv2`, data downloads, `git clone` |
| Session timeout | Max | long preprocessing/training steps |

> Phone-verified accounts get ~30 h GPU/week and up to 12 h per session
> (9 h without verification). A 100-epoch nnU-Net run on Task02_Heart or a
> 50-case BraTS subset fits in one session; full 1000-epoch runs do not —
> see the *resume* recipe below.

## 2. Getting the data onto Kaggle (3 options)

**Option A — Kaggle Datasets (best):** search and attach
- BraTS 2021: [`awsaf49/brats-2021-task-data`](https://www.kaggle.com/datasets/awsaf49/brats-2021-task-data)
- MSD tasks: several community mirrors exist (search "medical segmentation decathlon").
Attach via *Add Input* — mounted read-only at `/kaggle/input/<name>`.

**Option B — download inside the notebook:** notebook 01 does this for
Task02_Heart via `gdown` (Google Drive). Slowest but zero setup.

**Option C — upload your own dataset:** if you already downloaded MSD/BraTS
locally, create a private Kaggle Dataset and upload (the BraTS zip is ~12 GB;
Kaggle allows up to 100 GB per dataset).

## 3. Getting *this repo* onto Kaggle

Notebooks `git clone` `https://github.com/naveed1011/nnunet-medseg-benchmark` in their first
cells, so **push the repo before you start a GPU session**. If the repo is **private**, either:
- make it public (recommended — it's your portfolio piece), or
- zip the repo → upload as a private Kaggle Dataset → attach it and point
  `REPO_DIR` at `/kaggle/input/<zip-name>`.

> Kaggle secrets are not needed: the notebooks clone a public repo over HTTPS.

## 4. The env-var trap ⚠️

nnU-Net finds its folders via three environment variables:

```bash
nnUNet_raw, nnUNet_preprocessed, nnUNet_results
```

Kaggle does **not** persist env vars between sessions, and the folders they
point to (`/tmp/...` or your custom path) are wiped too. Every notebook here
therefore re-creates them under `/kaggle/working/nnunet_env/` in its setup cell.
**Only `/kaggle/working` survives** (as notebook output).

Two more things that only bite on Kaggle:

- `nnUNet_n_proc_DA`: the default (12) dataloader workers will fight over 4 vCPUs.
  The notebooks set it to 2.
- `nnUNet_compile`: torch.compile is slow to warm up and occasionally fails on these
  images; the repo's `run_pipeline.py` sets it to `False`.

## 5. Chaining the phases (the workflow that actually works)

```
02_nnunet_brats.ipynb ──Save & Run All──▶ Output v1
                                            │ attach as Input
03_swinunetr_brats_monai.ipynb ◀────────────┘  (needs brats_subset_split.json)
        │ Save & Run All ──▶ Output v2
                                │ attach BOTH outputs
04_compare_nnunet_vs_swin.ipynb ◀┘
05_xai_analysis.ipynb ◀── attach 03's output (swinunetr_best.pt) + 02's snapshot
```

After each notebook: **Save Version → Save & Run All**, wait for it to finish,
then in the next notebook use *Add Input → Your Work* to attach that output.

## 6. Surviving the 12 h limit

- **Snapshot results before the session ends**: notebook 02 has a snapshot cell
  that copies `nnUNet_results` (checkpoints, progress.png, validation) into
  `/kaggle/working/nnunet_snapshots/`. Output size limit is ~20 GB — the cell
  drops big optimizer-state files automatically.
- **Resume a dead training**: in a new session, re-run setup → convert →
  plan/preprocess, restore the snapshot into `$nnUNet_results`, then:
  ```bash
  nnUNetv2_train 701 3d_fullres 0 -tr nnUNetTrainer_100epochs --c
  ```
  (`--c` continues from `checkpoint_latest`). nnU-Net checkpoints every 50 epochs,
  so you lose at most ~50 epochs of work.
- **Reduce scope first, not quality second**: fewer cases (`MAX_CASES=30`)
  beats fewer epochs when you're pipeline-bound — the comparison stays fair as
  long as BOTH arms use the same subset.

## 7. Practical tips

- Watch training curves live: Kaggle file panel → `/kaggle/working/nnunet_env/res/.../fold_0/progress.png` (refreshes each epoch).
- `!nvidia-smi` in a cell tells you which GPU you actually got (T4 vs P100 matters for time estimates).
- Notebook outputs are private by default — attach them to other notebooks freely.
- If `pip install nnunetv2` upgrades torch and breaks the session's CUDA, add
  `--no-deps` and install missing deps one by one (rare, but it happens).
- Download artifacts you care about (figures, summary.json, CSVs) via the
  Output panel — don't rely on notebook outputs living forever.

## 8. Cost reality check

| Phase | GPU-hours (T4) | Notes |
|---|---|---|
| 01 Heart pipeline | ~4–6 h | mostly training |
| 02 BraTS nnU-Net (50 cases, 100 ep) | ~8–11 h | tight — start early in the session |
| 03 BraTS Swin UNETR (50 cases, 100 ep) | ~6–9 h | caching makes epochs fast |
| 04 Compare | < 0.5 h | CPU is enough |
| 05 XAI | ~1–2 h | CPU-OK for GradCAM, GPU nicer |

One full pass ≈ 20–28 GPU-hours ≈ one week of free quota. Plan accordingly
(run 01 while waiting for quota to refill for 02/03).
