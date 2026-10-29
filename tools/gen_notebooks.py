#!/usr/bin/env python3
"""Generate the Kaggle-ready notebooks for the nnunet-medseg-benchmark repo.

The notebooks are the artifact people run; this script is the source of truth
for their *text*, which makes bulk edits (a new repo URL, a fixed evaluation
protocol) reviewable in one place. Run it from anywhere:

    python tools/gen_notebooks.py
"""
import json
import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1] / "notebooks")
os.makedirs(ROOT, exist_ok=True)

def nb(path, cells):
    out = {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10"}},
           "nbformat": 4, "nbformat_minor": 5, "cells": cells}
    Path(path).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)

def md(s):  return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}
def code(s):
    # collapse doubled braces left over from f-string escaping in the generator
    s = s.replace("{{", "{").replace("}}", "}")
    return {"cell_type": "code", "metadata": {}, "source": s.strip().splitlines(keepends=True),
            "execution_count": None, "outputs": []}

REPO_PLACEHOLDER = "naveed1011/nnunet-medseg-benchmark"

# ============================================================================
# 01 — nnU-Net pipeline on MSD Task02_Heart (learn the framework)
# ============================================================================
nb(f"{ROOT}/01_nnunet_heart_msd.ipynb", [
md(f"""# 🫀 nnU-Net v2 end-to-end on MSD Task02_Heart — "Hello nnU-Net"

**Goal:** learn the complete nnU-Net v2 pipeline on the smallest Medical Segmentation
Decathlon task (left-atrium segmentation, 20 cases of 3D cardiac MRI), inside a single
free **Kaggle GPU session**.

| The 5 pipeline steps | Command |
|---|---|
| 1. Dataset in nnU-Net format | `Dataset002_Heart/` with `dataset.json` |
| 2. Fingerprint + planning + preprocessing | `nnUNetv2_plan_and_preprocess` |
| 3. Training | `nnUNetv2_train` |
| 4. Inference | `nnUNetv2_predict` |
| 5. Evaluation | `python -m src.metrics` (this repo) |

**⚙️ Before running:** Notebook Settings → Accelerator → **GPU T4 x2** (or P100), and
**Internet: On** (needed for pip + the Google Drive download).

**⏱️ Runtime budget:** install ~10 min · download ~10 min · planning/preprocessing ~5 min ·
training (100 epochs) ~3–6 h on T4 · predict+eval ~15 min. Fits one 12 h session with margin.

> Full 1000-epoch nnU-Net training is the SOTA recipe (~2 days on a good GPU). Here we use the
> built-in `nnUNetTrainer_100epochs` preset so everything fits in one session. The numbers are
> therefore *not* leaderboard numbers — the **pipeline understanding** is the point, and the same
> reduced budget is given to the Swin UNETR arm later, keeping the comparison fair.
"""),
md("## 0 · Install & environment"),
code("""!nvidia-smi -L
!pip install -q -U nnunetv2 gdown"""),
code("""import os, json, glob, shutil
BASE = "/kaggle/working/nnunet_env"
for k in ("raw", "pre", "res"):
    os.makedirs(f"{BASE}/{k}", exist_ok=True)
os.environ["nnUNet_raw"]          = f"{BASE}/raw"
os.environ["nnUNet_preprocessed"] = f"{BASE}/pre"
os.environ["nnUNet_results"]      = f"{BASE}/res"
# keeps a CPU-backed Kaggle kernel from spawning 12 dataloader workers
os.environ.setdefault("nnUNet_n_proc_DA", "2")
print("nnUNet_raw          =", os.environ["nnUNet_raw"])
print("nnUNet_preprocessed =", os.environ["nnUNet_preprocessed"])
print("nnUNet_results      =", os.environ["nnUNet_results"])"""),
code(f"""# Clone the repo (has src/metrics.py used for evaluation).
# >>> Replace with YOUR repo URL after you push it to GitHub <<<
!git clone --depth 1 https://github.com/{REPO_PLACEHOLDER}.git /kaggle/working/repo
# If the repo is still private: upload it as a Kaggle Dataset instead and set REPO_DIR below.
REPO_DIR = "/kaggle/working/repo"
os.chdir(REPO_DIR)
print(os.listdir("."))"""),
md("""## 1 · Download MSD Task02_Heart (~2 GB, Google Drive)

The Medical Segmentation Decathlon is hosted on Google Drive; `gdown` fetches it.
(Off-Kaggle you can do the same with `python data/download_msd.py --task Task02_Heart`.)"""),
code("""%%time
import gdown, tarfile, pathlib
TAR = "/kaggle/working/Task02_Heart.tar"
gdown.download(id="1wEB2I6S6tQBVEPxir8cB7kFB8gTQiQiT", path=TAR, quiet=False, fuzzy=True)
with tarfile.open(TAR) as tf:
    try:
        tf.extractall("/kaggle/working", filter="data")
    except TypeError:          # Python < 3.12
        tf.extractall("/kaggle/working")
task = pathlib.Path("/kaggle/working/Task02_Heart")
print("cases:", len(list((task/"imagesTr").glob("*.nii.gz"))))"""),
md("""## 2 · Convert to nnU-Net v2 format

The MSD folder *looks* like an nnU-Net dataset, but its `dataset.json` uses the **v1 key
`modality`**, while nnU-Net v2 hard-requires **`channel_names`** — copying it as-is makes
`plan_and_preprocess` fail at the integrity check. The repo's converter rewrites
`dataset.json` in the v2 schema (`channel_names`, `file_ending`, consecutive labels) and
copies the 20 cases into `Dataset002_Heart`."""),
code("""%%time
# The repo was cloned in the setup cell; the converter rewrites dataset.json in
# the v2 schema (channel_names instead of MSD's v1 'modality').
!cd {REPO_DIR} && python data/convert_to_nnunet.py --task heart --raw /kaggle/working/Task02_Heart --out $nnUNet_raw --copy

dj = json.load(open(os.path.join(os.environ["nnUNet_raw"], "Dataset002_Heart", "dataset.json")))
print("channels:", dj["channel_names"])
print("labels:  ", dj["labels"])
print("numTraining:", dj["numTraining"])"""),
md("""## 3 · Experiment planning & preprocessing

This is where nnU-Net's magic happens: it extracts a **dataset fingerprint** (image sizes,
voxel spacings, intensity statistics) and *automatically* decides patch size, batch size,
target spacing, normalization scheme, network depth and more. Watch what it chooses —
reading `nnUNetPlans.json` afterwards is one of the best ways to understand the framework."""),
code("""%%time
!nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity"""),
code("""# 🔍 Inspect what nnU-Net planned for us
import json, glob
plans_path = glob.glob(os.path.join(os.environ["nnUNet_preprocessed"], "Dataset002_Heart", "nnUNetPlans.json"))[0]
plans = json.load(open(plans_path))
cfg = plans["configurations"]["3d_fullres"]
for key in ("patch_size", "batch_size", "spacing", "normalization_schemes", "n_conv_per_stage_encoder", "n_conv_per_stage_decoder"):
    print(f"{key:>28} : {cfg.get(key)}")
print("\\nAvailable configurations:", list(plans["configurations"].keys()))"""),
md("""## 4 · Training (fold 0, 3d_fullres, 100-epoch preset)

nnU-Net's default is 1000 "epochs" (250 iterations each) with 5-fold cross-validation.
For a Kaggle session we train **fold 0 only, 100 epochs** via the built-in reduced trainer.
`--npz` additionally stores softmax outputs of the final validation (needed later for
ensembling / `find_best_configuration`).

📈 While it runs, you can open `progress.png` inside
`$nnUNet_results/Dataset002_Heart/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/fold_0/`
from the Kaggle file panel — it updates after every epoch."""),
code("""%%time
!nnUNetv2_train 2 3d_fullres 0 -tr nnUNetTrainer_100epochs --npz"""),
md("""## 5 · Inference on the **validation** cases

Only the 4 held-out cases of fold 0. Predicting the 16 training cases as well would make the
evaluation look better than the model is; the held-out slice is the number worth quoting, and
it is the same protocol the BraTS arm uses later."""),
code("""%%time
RAW = os.path.join(os.environ["nnUNet_raw"], "Dataset002_Heart")
PRED_DIR = "/kaggle/working/predictions/heart_fold0"
VAL_IN   = "/kaggle/working/val_images_heart"
splits = json.load(open(os.path.join(os.environ["nnUNet_preprocessed"], "Dataset002_Heart", "splits_final.json")))
val_cases = splits[0]["val"]
os.makedirs(VAL_IN, exist_ok=True); os.makedirs(PRED_DIR, exist_ok=True)
for c in val_cases:
    for f in glob.glob(os.path.join(RAW, "imagesTr", f"{c}_*.nii.gz")):
        shutil.copy2(f, VAL_IN)
print("held-out cases:", val_cases)
!nnUNetv2_predict -i {VAL_IN} -o {PRED_DIR} -d 2 -c 3d_fullres -f 0
print(sorted(os.listdir(PRED_DIR)))"""),
md("""## 6 · Evaluate with this repo's metric module

Dice + HD95 (in mm!) on the 4 held-out volumes — the exact same code path the BraTS
comparison uses, so the two arms are not measured with two different rulers."""),
code("""%%time
import sys; sys.path.insert(0, REPO_DIR)
from src.metrics import evaluate_folders
from pathlib import Path
GT = os.path.join(RAW, "labelsTr")
report = evaluate_folders(PRED_DIR, GT, labels=(1,))
for k, v in report.mean.items():
    print(k, "->", {kk: round(vv, 4) for kk, vv in v.items()})
report.save(Path("/kaggle/working/results/heart_nnunet"))"""),
code("""# nnU-Net's OWN validation summary — it should agree with the numbers above
import glob
summ = glob.glob(os.path.join(os.environ["nnUNet_results"], "Dataset002_Heart", "*", "fold_0", "validation", "summary.json"))
if summ:
    s = json.load(open(summ[0]))
    print("mean:", json.dumps(s.get("mean", {}), indent=2)[:1200])
    print("full summary at:", summ[0])"""),
md("""## 7 · Package outputs for persistence

Anything in `/kaggle/working` is saved when you **Save Version → Save & Run All**, and can be
attached to later notebooks as a Dataset. We copy the artifacts that matter (metrics, curves,
plans, a few predictions) and zip them — never the raw data."""),
code("""%%time
out = "/kaggle/working/outputs_heart"
shutil.rmtree(out, ignore_errors=True); os.makedirs(out, exist_ok=True)
shutil.copytree("/kaggle/working/results/heart_nnunet", f"{out}/metrics", dirs_exist_ok=True)
for f in glob.glob(os.path.join(os.environ["nnUNet_results"], "Dataset002_Heart/**/progress.png"), recursive=True):
    shutil.copy2(f, f"{out}/progress.png")
for f in glob.glob(os.path.join(os.environ["nnUNet_results"], "Dataset002_Heart/**/validation/summary.json"), recursive=True):
    shutil.copy2(f, f"{out}/nnunet_validation_summary.json")
shutil.copy2(os.path.join(os.environ["nnUNet_preprocessed"], "Dataset002_Heart", "nnUNetPlans.json"),
             f"{out}/nnUNetPlans.json")
shutil.copytree(VAL_IN, f"{out}/val_images", dirs_exist_ok=True)
shutil.copytree(PRED_DIR, f"{out}/predictions_val", dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/nnunet_heart_outputs", "zip", out)
print("📦 zipped -> outputs_heart.zip")"""),
md("""## ✅ Done — what you learned

1. nnU-Net v2 dataset format (`DatasetXXX_Name`, `dataset.json`, `_0000` channel suffixes)
2. How the **fingerprint → plans** step auto-configures patch size / batch size / spacing
3. Training with reduced-epoch trainers under a compute budget (and what `--npz` is for)
4. Sliding-window inference with `nnUNetv2_predict`
5. Independent evaluation (Dice + HD95) with this repo's `src/metrics.py`

**Next:** `02_nnunet_brats.ipynb` — same pipeline, real research question:
brain tumor segmentation on BraTS, with a shared data split for the Swin UNETR comparison.
"""),
])

# ============================================================================
# 02 — nnU-Net on BraTS 2021 (research arm)
# ============================================================================
nb(f"{ROOT}/02_nnunet_brats.ipynb", [
md(f"""# 🧠 nnU-Net v2 on BraTS 2021 — brain tumor segmentation (research arm)

**Goal:** run nnU-Net on a BraTS 2021 subset with a **fixed, documented data split** that the
Swin UNETR arm (notebook 03) will reuse — this is what turns "running a framework" into a
**controlled comparison study**.

**📎 Required Kaggle Dataset:** [`awsaf49/brats-2021-task-data`](https://www.kaggle.com/datasets/awsaf49/brats-2021-task-data)
(Add Input → search "brats 2021 task data"). It mirrors the official RSNA-ASNR-MICCAI BraTS 2021
training data (1251 cases, 4 MRI modalities + segmentation).

**⚙️ Settings:** GPU T4 x2 (or better) · Internet On.

**⏱️ Budget (50 cases, 100 epochs, T4):** convert ~10 min · plan/preprocess ~30–60 min ·
train ~6–9 h · predict+eval ~20 min. **Tight but fits 12 h.** Scale knobs:
`MAX_CASES` (↓ faster) and `-tr nnUNetTrainer_250epochs` (↑ closer to SOTA, needs A100 or 2 sessions).

### BraTS label mapping (documented for reproducibility)
| Original value | Structure | Remapped value (this repo) |
|---|---|---|
| 0 | background | 0 |
| 1 | necrotic & non-enhancing tumor (NCR/NET) | 1 |
| 2 | peritumoral edema (ED) | 2 |
| 4 | gadolinium-enhancing tumor (ET) | **3** ← nnU-Net needs consecutive labels |
"""),
md("## 0 · Setup"),
code("""!nvidia-smi -L
!pip install -q -U nnunetv2"""),
code("""import os, glob, json, pathlib
BASE = "/kaggle/working/nnunet_env"
for k in ("raw", "pre", "res"):
    os.makedirs(f"{BASE}/{k}", exist_ok=True)
os.environ["nnUNet_raw"]          = f"{BASE}/raw"
os.environ["nnUNet_preprocessed"] = f"{BASE}/pre"
os.environ["nnUNet_results"]      = f"{BASE}/res"

MAX_CASES   = 50     # <-- scale knob: 50 fits one T4 session; 100+ needs A100 / 2 sessions
EPOCHS      = 100    # nnUNetTrainer_{EPOCHS}epochs preset (built-in: 50/100/250/500)
DATASET_ID  = 701
CONFIG      = "3d_fullres"

# locate the attached BraTS dataset (name may vary slightly with the Kaggle mount)
cands = glob.glob("/kaggle/input/*brats*2021*") + glob.glob("/kaggle/input/*brats*")
assert cands, "Attach the Kaggle dataset 'awsaf49/brats-2021-task-data' first!"
BRATS_ROOT = cands[0]
train_dirs = glob.glob(f"{BRATS_ROOT}/**/RSNA_ASNR_MICCAI_BraTS_2021_TrainingData", recursive=True)
TRAIN_DIR = train_dirs[0] if train_dirs else BRATS_ROOT
cases = sorted(d for d in glob.glob(f"{TRAIN_DIR}/BraTS2021_*") if os.path.isdir(d))
print(f"BraTS root: {BRATS_ROOT}\\n{len(cases)} total cases — using first {MAX_CASES}")"""),
code(f"""!git clone --depth 1 https://github.com/{REPO_PLACEHOLDER}.git /kaggle/working/repo
REPO_DIR = "/kaggle/working/repo"
import sys; sys.path.insert(0, REPO_DIR)"""),
md("""## 1 · Fixed split FIRST (so nnU-Net and Swin UNETR see identical data)

We take the first `MAX_CASES` cases in sorted order, split **80/20 deterministic** (every 5th
case → validation), and inject this split into nnU-Net via its `splits_final.json` mechanism.
The same split file is exported to `/kaggle/working` so notebook 03 reuses it verbatim.
*Deterministic-by-design beats random here: anyone can reproduce it without a seed hunt.*"""),
code("""case_ids_all = [os.path.basename(d) for d in cases[:MAX_CASES]]
train_ids = [c for i, c in enumerate(case_ids_all) if i % 5 != 4]
val_ids   = [c for i, c in enumerate(case_ids_all) if i % 5 == 4]
split = {"task": "BraTS2021", "max_cases": MAX_CASES, "note": "new ids are BraTS21_%05d by sorted original order",
         "id_map": {f"BraTS21_{i:05d}": c for i, c in enumerate(case_ids_all, 1)},
         "train": [f"BraTS21_{i:05d}" for i, c in enumerate(case_ids_all, 1) if (i-1) % 5 != 4],
         "val":   [f"BraTS21_{i:05d}" for i, c in enumerate(case_ids_all, 1) if (i-1) % 5 == 4]}
pathlib.Path("/kaggle/working/split").mkdir(exist_ok=True)
json.dump(split, open("/kaggle/working/split/brats_subset_split.json", "w"), indent=2)
print(f"train: {{len(split['train'])}}   val: {{len(split['val'])}}")
print("val cases:", split["val"])"""),
md("""## 2 · Convert to nnU-Net format (`Dataset701_BraTS21`)

The converter (from this repo) renames modalities to nnU-Net channel suffixes
(`flair,t1,t1ce,t2 → _0000.._0003`), **remaps label 4 → 3**, and writes a proper
`dataset.json`. Files are copied (not symlinked) so everything lives under `/kaggle/working`.""",),
code("""%%time
!cd {REPO_DIR} && python data/convert_to_nnunet.py --task brats --raw {BRATS_ROOT} --max-cases {MAX_CASES} --copy --out $nnUNet_raw"""),
code("""# NOTE: nnU-Net creates the preprocessed folder during plan_and_preprocess (next section),
# so we inject the fixed split AFTER planning and BEFORE training.
print("split file ready at /kaggle/working/split/brats_subset_split.json")"""),
md("""## 3 · Planning & preprocessing (3d_fullres only — Kaggle budget)

With 4 modalities at 1 mm isotropic BraTS resolution, preprocessing is the slowest step after
training. `-c 3d_fullres` skips the 2d config (brain tumors are inherently 3D structures, and
this also saves session time — a deliberate, documented choice)."""),
code("""%%time
!nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity -c 3d_fullres -np 4"""),
code("""# NOW inject the fixed split (nnU-Net created the preprocessed folder above).
# nnU-Net v2 reads `splits_final.json` (a list of {"train": [...], "val": [...]});
# fold 0 becomes our fixed split, so both model arms see identical cases.
pre_dir = os.path.join(os.environ["nnUNet_preprocessed"], f"Dataset{DATASET_ID}_BraTS21")
splits = [{"train": split["train"], "val": split["val"]}]
json.dump(splits, open(os.path.join(pre_dir, "splits_final.json"), "w"), indent=2)
print("injected fold_0 split:", len(splits[0]["train"]), "train /", len(splits[0]["val"]), "val")"""),
code("""# 🔍 What did nnU-Net plan for BraTS?
plans = json.load(open(os.path.join(pre_dir, "nnUNetPlans.json")))
cfg = plans["configurations"]["3d_fullres"]
for key in ("patch_size", "batch_size", "spacing", "normalization_schemes", "resampling_fn_data"):
    print(f"{key:>24} : {cfg.get(key)}")"""),
md("""## 4 · Train fold 0 (the fixed split)

`--npz` stores validation softmax outputs. Checkpoints are written every 50 epochs
(`checkpoint_latest.*`) — the snapshot cell below protects them against Kaggle session loss."""),
code("""%%time
!nnUNetv2_train {DATASET_ID} {CONFIG} 0 -tr nnUNetTrainer_{EPOCHS}epochs --npz"""),
md("""### 💾 Snapshot results (run this BEFORE the session ends!)

`/kaggle/working` persists as notebook output; the env folders do not. Copy checkpoints +
progress + validation so you can (a) download them and (b) attach them to notebooks 04/05.
If a session dies mid-training: re-run setup+convert+preprocess, restore the snapshot into
`$nnUNet_results`, and continue with `nnUNetv2_train ... --c`.""",),
code("""%%time
import shutil
res_dir = os.path.join(os.environ["nnUNet_results"], f"Dataset{DATASET_ID}_BraTS21")
snap = f"/kaggle/working/nnunet_snapshots/Dataset{DATASET_ID}_BraTS21"
if os.path.exists(snap): shutil.rmtree(snap)
shutil.copytree(res_dir, snap)
print("snapshot size: %.1f MB" % (sum(os.path.getsize(os.path.join(dp, fn))
      for dp, _, fs in os.walk(snap) for fn in fs) / 1024**2))
print("kept checkpoint_latest so a dead session can resume with --c")"""),
md("## 5 · Predictions on the validation cases + evaluation"),
code("""%%time
# build an input folder with ONLY the validation cases (4 channels each)
val_in = "/kaggle/working/val_images_brats"; os.makedirs(val_in, exist_ok=True)
img_dir = os.path.join(os.environ["nnUNet_raw"], f"Dataset{DATASET_ID}_BraTS21", "imagesTr")
for cid in split["val"]:
    for f in glob.glob(os.path.join(img_dir, f"{cid}_*.nii.gz")):
        shutil.copy2(f, val_in)
print(len(os.listdir(val_in)), "files (", len(split["val"]), "cases x 4 modalities )")
!nnUNetv2_predict -i {val_in} -o /kaggle/working/predictions/brats_nnunet_val -d {DATASET_ID} -c {CONFIG} -f 0"""),
code("""%%time
from src.metrics import evaluate_folders
gt_dir = os.path.join(os.environ["nnUNet_raw"], f"Dataset{DATASET_ID}_BraTS21", "labelsTr")
report = evaluate_folders("/kaggle/working/predictions/brats_nnunet_val", gt_dir, labels=(1, 2, 3))
for k, v in report.mean.items():
    print(f"{k:>10} ->", {kk: round(vv, 4) for kk, vv in v.items()})
report.save(pathlib.Path("/kaggle/working/results/brats_nnunet"))"""),
code("""# package everything for the next notebooks (attach as Kaggle Dataset output)
import shutil
out = "/kaggle/working/outputs_brats"
os.makedirs(out, exist_ok=True)
for f in glob.glob("/kaggle/working/results/brats_nnunet/*"): shutil.copy2(f, out)
shutil.copy2("/kaggle/working/split/brats_subset_split.json", out)
for f in glob.glob(f"{snap}/**/progress.png", recursive=True): shutil.copy2(f, f"{out}/progress_nnunet.png")
for f in glob.glob(f"{snap}/**/validation/summary.json", recursive=True): shutil.copy2(f, f"{out}/nnunet_validation_summary.json")
for f in glob.glob(f"{snap}/**/plans.json", recursive=True): shutil.copy2(f, f"{out}/nnunet_plans.json")
print("📦 outputs:", os.listdir(out))"""),
md("""## ✅ Results checklist & honest notes

- [ ] `results/brats_nnunet/summary.json` — Dice/HD95 per label (NCR, ED, ET)
- [ ] `progress.png` — loss + pseudo-dice curves (copy into your repo's `docs/figures/`!)
- [ ] `brats_subset_split.json` — the shared split for notebook 03
- [ ] snapshot of `nnUNet_results` (for XAI in notebook 05)

**Write these caveats into your report** (professors respect this):
1. Subset size (MAX_CASES) & reduced epoch budget (EPOCHS) → numbers are *pipeline-validation
   quality*, not BraTS-challenge quality. State it plainly.
2. Predictions on val cases come from a model that never saw them — that's the honest number;
   ignore train-case predictions.
3. HD95 = ∞ when a structure is missed entirely (esp. small ET on some cases) — we report the
   % of inf cases rather than silently dropping them.
"""),
])

# ============================================================================
# 03 — Swin UNETR arm (MONAI), same split
# ============================================================================
nb(f"{ROOT}/03_swinunetr_brats_monai.ipynb", [
md(f"""# 🦢 Swin UNETR on the SAME BraTS subset — the comparison arm

**Goal:** train MONAI's **Swin UNETR** on *exactly* the same data split as the nnU-Net arm
(notebook 02) and evaluate with *exactly* the same metrics — a controlled comparison.

**📎 Required Kaggle Datasets:**
1. [`awsaf49/brats-2021-task-data`](https://www.kaggle.com/datasets/awsaf49/brats-2021-task-data)
2. **The output of notebook 02** (Save Version on it first, then Add Input → your notebook output)
   — we need `brats_subset_split.json` from it.

**⚙️ Settings:** GPU T4 x2 · Internet On.

### Fair-comparison protocol (this is the research core — keep it identical!)
| Knob | nnU-Net arm | Swin UNETR arm |
|---|---|---|
| Cases | first {50} sorted, 80/20 split | **same file** `brats_subset_split.json` |
| Preprocessing | 1 mm iso resample, per-channel foreground z-score (nnU-Net CT-independent scheme) | **matched manually below** |
| Budget | `nnUNetTrainer_100epochs`, fold 0 | 100 epochs, fold 0 split, similar patch volume |
| Metrics | `src/metrics.py` (Dice, HD95-mm) | **identical code path** |

> ⚠️ Honest limitation to state in your writeup: nnU-Net additionally auto-tunes patch/batch
> size, augmentation and architecture per dataset; we give Swin UNETR sensible fixed defaults
> instead of an equally tuned setup. The comparison is *budget-matched*, not *tuning-matched*.
"""),
md("## 0 · Setup & data"),
code("""!pip install -q -U "monai[nibabel]>=1.3"
import os, glob, json, pathlib, numpy as np, pandas as pd
SPLIT_FILES = glob.glob("/kaggle/input/*/brats_subset_split.json") + glob.glob("/kaggle/working/**/brats_subset_split.json", recursive=True)
assert SPLIT_FILES, "Attach the OUTPUT of notebook 02 as a Kaggle Dataset (it contains brats_subset_split.json)!"
split = json.load(open(SPLIT_FILES[0]))
print("split file:", SPLIT_FILES[0])
print("train:", len(split["train"]), " val:", len(split["val"]))"""),
code(f"""!git clone --depth 1 https://github.com/{REPO_PLACEHOLDER}.git /kaggle/working/repo 2>/dev/null || true
import sys; sys.path.insert(0, "/kaggle/working/repo")"""),
code("""# map BraTS21_XXXXX -> original case folder (notebook 02 recorded id_map)
cands = glob.glob("/kaggle/input/*brats*")
TRAIN_DIRS = glob.glob(f"{cands[0]}/**/RSNA_ASNR_MICCAI_BraTS_2021_TrainingData", recursive=True) if cands else []
TRAIN_DIR = TRAIN_DIRS[0] if TRAIN_DIRS else cands[0]
id_map = split["id_map"]   # {{"BraTS21_00001": "BraTS2021_00xxx", ...}}
orig_of = {{k: os.path.join(TRAIN_DIR, v) for k, v in id_map.items()}}
missing = [k for k, v in orig_of.items() if not os.path.isdir(v)]
assert not missing, f"{{len(missing)}} case folders not found — is the BraTS dataset attached? e.g. {{missing[:3]}} (looked in {{TRAIN_DIR}})"
print("✅ all", len(orig_of), "case folders resolved")"""),
md("""## 1 · Dataset class — preprocessing matched to nnU-Net's BraTS scheme

BraTS scans are already skull-stripped, co-registered and 1 mm isotropic, so resampling is a
no-op (we assert it). Intensity handling follows nnU-Net's default for MRI-like data:
**clip to foreground percentiles [0.5, 99.5], then z-score with foreground mean/std** — per channel."""),
code("""import torch, nibabel as nib
from collections import OrderedDict
from torch.utils.data import Dataset

MODS = ("flair", "t1", "t1ce", "t2")
# BraTS {0,1,2,4} -> consecutive {0,1,2,3}; a lookup table beats np.vectorize here
_REMAP_LUT = np.array([0, 1, 2, 0, 3], dtype=np.int8)

def load_case(new_id):
    d = orig_of[new_id]
    imgs = []
    for m in MODS:
        hits = [f for f in glob.glob(os.path.join(d, f"*{m}.nii.gz")) if "seg" not in os.path.basename(f).lower()]
        imgs.append(nib.load(hits[0]))
    seg = nib.load(glob.glob(os.path.join(d, "*seg.nii.gz"))[0])
    X = np.stack([np.asarray(im.dataobj, dtype=np.float32) for im in imgs])   # (4, D, H, W)
    y = _REMAP_LUT[np.clip(np.asarray(seg.dataobj).astype(np.int16), 0, 4)]
    return X, y, imgs[0].affine, new_id

def normalize(x):
    fg = x[x > 0]
    if fg.size == 0: return x
    lo, hi = np.percentile(fg, [0.5, 99.5])
    x = np.clip(x, lo, hi)
    fg = x[x > 0]
    return (x - fg.mean()) / (fg.std() + 1e-8)

class BraTSSubset(Dataset):
    \"\"\"Random 96^3 crops for training.

    The cache is bounded on purpose: a BraTS volume is ~36 MB as float32 and two
    DataLoader workers would otherwise keep every case in RAM (13 GB on Kaggle).
    \"\"\"
    def __init__(self, ids, augment=False, cache_size=4):
        self.ids, self.augment = list(ids), augment
        self.cache = OrderedDict()
        self.cache_size = cache_size
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        cid = self.ids[i]
        if cid not in self.cache:
            X, y, aff, _ = load_case(cid)
            assert X.shape[1:] == (240, 240, 155), f"unexpected BraTS shape {X.shape} — data not 1mm iso?"
            X = np.stack([normalize(c) for c in X])
            self.cache[cid] = (X.astype(np.float32), y)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        X, y = self.cache[cid]
        X, y = X.copy(), y.copy()
        if self.augment:
            if np.random.rand() < 0.5: X, y = X[..., ::-1].copy(), y[..., ::-1].copy()
            if np.random.rand() < 0.5: X, y = X[:, :, ::-1].copy(), y[:, :, ::-1].copy()
            X = X + np.random.normal(0, 0.01, X.shape).astype(np.float32)
        # random crop 96^3 (same volume ballpark as nnU-Net's auto-planned patch)
        D, H, W = X.shape[1:]
        ps = 96
        d0 = np.random.randint(0, D - ps); h0 = np.random.randint(0, H - ps); w0 = np.random.randint(0, W - ps)
        Xc = X[:, d0:d0+ps, h0:h0+ps, w0:w0+ps]
        yc = y[d0:d0+ps, h0:h0+ps, w0:w0+ps]
        # ensure the crop contains foreground ~80% of the time (else re-center on a lesion)
        if yc.max() == 0 and np.random.rand() < 0.8:
            fg = np.argwhere(y > 0)
            if len(fg):
                c = fg[np.random.randint(len(fg))]
                d0 = int(np.clip(c[0]-ps//2, 0, D-ps)); h0 = int(np.clip(c[1]-ps//2, 0, H-ps)); w0 = int(np.clip(c[2]-ps//2, 0, W-ps))
                Xc = X[:, d0:d0+ps, h0:h0+ps, w0:w0+ps]; yc = y[d0:d0+ps, h0:h0+ps, w0:w0+ps]
        return torch.from_numpy(Xc), torch.from_numpy(yc.astype(np.int64))

train_ds = BraTSSubset(split["train"], augment=True)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
print(f"train batches/epoch: {len(train_dl)}   val cases: {len(split['val'])}")"""),
md("## 2 · Model — MONAI Swin UNETR (pretrained on BTCV, 4 in-channels, 4 classes)"),
code("""from monai.networks.nets import SwinUNETR
device = torch.device("cuda")
model = SwinUNETR(img_size=(96, 96, 96), in_channels=4, out_channels=4,
                  feature_size=24, use_v2=True).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"SwinUNETR params: {{n_params/1e6:.1f}}M")"""),
md("""## 3 · Training — 100 epochs, DiceCE loss, cosine LR

**Budget honesty:** nnU-Net's "100 epochs" means 100 × 250 iterations ≈ 25k optimizer
steps. One Swin epoch here is a single pass over the 40 training volumes (~40 steps), so the
two arms are matched on *epochs*, not on *iterations*. I state that in the results instead of
pretending the budgets are identical."""),
code("""from monai.inferers import sliding_window_inference

def dice_per_class(pred, gt, n_classes=4):
    out = []
    for c in range(1, n_classes):
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        out.append(1.0 if denom == 0 else float(2 * (p & g).sum() / denom))
    return out

@torch.no_grad()
def validate_full(model, ids, roi=(96, 96, 96)):
    \"\"\"Dice on the FULL held-out volumes — not on random 96^3 crops.

    Validating on random crops (the easy shortcut) reports a number that depends
    on where the crop happened to land. This is the honest version, and it is the
    only one worth putting next to nnU-Net's validation Dice.
    \"\"\"
    model.eval()
    scores = []
    for cid in ids:
        X, y, _, _ = load_case(cid)
        Xn = np.stack([normalize(c) for c in X]).astype(np.float32)
        logits = sliding_window_inference(torch.from_numpy(Xn[None]).to(device),
                                          roi_size=roi, sw_batch_size=1, predictor=model, overlap=0.5)
        pred = torch.argmax(torch.softmax(logits, 1), 1)[0].cpu().numpy()
        scores.append(float(np.mean(dice_per_class(pred, y))))
    model.train()
    return float(np.mean(scores))"""),
code("""%%time
from monai.losses import DiceCELoss
import time

EPOCHS = 100
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
loss_fn = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True)

best_dice, history = -1, []
for epoch in range(EPOCHS):
    model.train(); t0 = time.time(); ep_loss = 0.0
    for X, y in train_dl:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        optimizer.step(); ep_loss += loss.item()
    scheduler.step()
    ep_loss /= len(train_dl)

    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:      # full-volume val every 10 epochs
        val = validate_full(model, split["val"])
        history.append({"epoch": epoch+1, "train_loss": round(ep_loss, 4),
                        "val_dice": round(val, 4), "lr": scheduler.get_last_lr()[0]})
        print(f"epoch {epoch+1:3d} | loss {ep_loss:.4f} | val_dice {val:.4f} | {time.time()-t0:.0f}s/epoch")
        if val > best_dice:
            best_dice = val
            torch.save({"model": model.state_dict(), "epoch": epoch+1,
                        "val_dice": val, "history": history}, "/kaggle/working/swinunetr_best.pt")
    else:
        history.append({"epoch": epoch+1, "train_loss": round(ep_loss, 4)})
print("✅ best val dice:", best_dice)"""),
md("## 4 · Inference on validation cases → NIfTI + evaluation (identical code path)"),
code("""%%time
import nibabel as nib
# weights_only=False: the checkpoint also carries the plain-python training history
ckpt = torch.load("/kaggle/working/swinunetr_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"]); model.eval()
out_dir = "/kaggle/working/predictions/brats_swin_val"; os.makedirs(out_dir, exist_ok=True)
t_inf = []
with torch.no_grad():
    for cid in split["val"]:
        X, y, aff, _ = load_case(cid)
        Xn = np.stack([normalize(c) for c in X]).astype(np.float32)
        t0 = time.time()
        logits = sliding_window_inference(torch.from_numpy(Xn[None]).to(device), roi_size=(96,96,96),
                                          sw_batch_size=2, predictor=model, overlap=0.5)
        pred = torch.argmax(torch.softmax(logits, 1), 1)[0].cpu().numpy().astype(np.uint8)
        t_inf.append(time.time() - t0)
        nib.save(nib.Nifti1Image(pred, aff), os.path.join(out_dir, f"{cid}.nii.gz"))
peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
print(f"mean inference time/volume: {np.mean(t_inf):.1f}s   peak GPU mem: {peak:.0f} MB")"""),
code("""%%time
import sys; sys.path.insert(0, "/kaggle/working/repo" if os.path.isdir("/kaggle/working/repo") else ".")
# metrics module: clone repo OR it's attached with notebook 02's output; fallback: pip-installed copy
try:
    from src.metrics import evaluate_folders
except ImportError:
    !git clone --depth 1 https://github.com/""" + REPO_PLACEHOLDER + """.git /kaggle/working/repo
    sys.path.insert(0, "/kaggle/working/repo")
    from src.metrics import evaluate_folders

# GT folder in the SAME remapped label space: build it from the raw labels
gt_dir = "/kaggle/working/gt_brats_remapped"; os.makedirs(gt_dir, exist_ok=True)
for cid in split["val"]:
    if not os.path.exists(f"{gt_dir}/{cid}.nii.gz"):
        _, y, aff, _ = load_case(cid)
        nib.save(nib.Nifti1Image(y.astype(np.uint8), aff), f"{gt_dir}/{cid}.nii.gz")
report = evaluate_folders(out_dir, gt_dir, labels=(1, 2, 3))
for k, v in report.mean.items():
    print(f"{k:>10} ->", {kk: round(vv, 4) for kk, vv in v.items()})
report.save(pathlib.Path("/kaggle/working/results/brats_swinunetr"))"""),
code("""# model stats for the efficiency part of the comparison
import json
stats = {"model": "SwinUNETR (MONAI, feature_size=24, v2)", "n_params_M": round(n_params/1e6, 2),
         "epochs": EPOCHS, "patch": 96, "batch": 1,
         "mean_inference_s_per_volume": round(float(np.mean(t_inf)), 1),
         "peak_gpu_mem_MB": round(float(peak), 0),
         "best_val_dice": round(float(ckpt["val_dice"]), 4)}
json.dump(stats, open("/kaggle/working/results/brats_swinunetr/model_stats.json", "w"), indent=2)
pd.DataFrame(ckpt["history"]).to_csv("/kaggle/working/results/brats_swinunetr/training_history.csv", index=False)
print(json.dumps(stats, indent=2))"""),
md("""## ✅ What this notebook produced
- `predictions/brats_swin_val/` — Swin UNETR predictions on the shared val split
- `results/brats_swinunetr/` — per-case CSV + summary.json (same format as the nnU-Net arm)
- `swinunetr_best.pt` — checkpoint (needed for XAI in notebook 05)
- `training_history.csv` + `model_stats.json` — learning curve & efficiency numbers

**Next:** notebook 04 merges both arms into the comparison table + figures.
"""),
])

# ============================================================================
# 04 — Comparison notebook
# ============================================================================
nb(f"{ROOT}/04_compare_nnunet_vs_swin.ipynb", [
md(f"""# ⚖️ nnU-Net vs Swin UNETR — the comparison study

**📎 Required Kaggle Datasets (attach BOTH):**
1. Output of **notebook 02** → contains `per_case_metrics.csv` / `summary.json` (nnU-Net arm)
2. Output of **notebook 03** → contains `per_case_metrics.csv` / `summary.json` (Swin arm)

This notebook builds:
- `comparison_table.md/.csv` — Dice & HD95 per label per model (mean ± std, median)
- `boxplot_per_case_dice.png` — per-case distributions (means hide variance — plots don't)
- a short **findings template** you should fill in by hand for your repo's writeup
"""),
code(f"""!git clone --depth 1 https://github.com/{REPO_PLACEHOLDER}.git /kaggle/working/repo
import os, glob, sys, json
sys.path.insert(0, "/kaggle/working/repo")
os.chdir("/kaggle/working/repo")"""),
code("""# locate the two arms' results among attached inputs. Both write
# per_case_metrics.csv, and "swinunetr" contains "nnunet" — so match the
# immediate parent folder name exactly instead of doing a substring search.
import os, glob
csvs = glob.glob("/kaggle/input/**/per_case_metrics.csv", recursive=True) + \\
       glob.glob("/kaggle/working/**/per_case_metrics.csv", recursive=True)
print("found result CSVs:")
for c in csvs: print("  ", c)

def pick(name):
    hits = [os.path.dirname(c) for c in csvs
            if os.path.basename(os.path.dirname(c)).lower() == name.lower()]
    assert hits, (f"No results folder named '{name}' — attach both notebook outputs "
                  f"(found: {[os.path.dirname(c) for c in csvs]})")
    return hits[0]

NNUNET_DIR = pick("brats_nnunet")
SWIN_DIR   = pick("brats_swinunetr")
assert NNUNET_DIR != SWIN_DIR, "both arms resolved to the same folder — fix the paths below"
print("nnU-Net arm:", NNUNET_DIR)
print("Swin arm:   ", SWIN_DIR)"""),
code("""%%time
# If the two CSVs are ambiguous by name, point the paths manually:
# NNUNET_DIR = "/kaggle/input/<nb02-output>/results/brats_nnunet"
# SWIN_DIR   = "/kaggle/input/<nb03-output>/results/brats_swinunetr"
!python -m src.compare --arm nnunet={NNUNET_DIR} --arm swinunetr={SWIN_DIR} --out /kaggle/working/comparison --task "BraTS2021 subset (NCR/ED/ET)"
print(open("/kaggle/working/comparison/comparison_table.md").read())"""),
code("""from IPython.display import Image, display
display(Image("/kaggle/working/comparison/figures/boxplot_per_case_dice.png"))
display(Image("/kaggle/working/comparison/figures/bar_dice_by_label.png"))"""),
md("""## 5 · Package the comparison for the repo

`src.compare` writes `RESULTS.md` next to the figures, so the repo's results page and the
Kaggle output cannot drift apart. Copy this folder into `docs/results/brats/` when you push."""),
code("""import shutil
out = "/kaggle/working/outputs_comparison"
shutil.rmtree(out, ignore_errors=True)
shutil.copytree("/kaggle/working/comparison", out)
shutil.copy2("/kaggle/working/split/brats_subset_split.json", out)
print("📦 outputs_comparison:", os.listdir(out))
print(open(f"{out}/RESULTS.md").read()[:1500])"""),
md("""## 📝 Findings template (fill this in — this is the research contribution!)

> **Setup.** Both models trained on the same {N_train}-case BraTS 2021 subset (80/20 fixed
> split), 100-epoch budget, evaluated on the same {N_val} held-out volumes with Dice + HD95-mm.
> nnU-Net self-configured: patch {…}, batch {…}, spacing {…} (read from `nnUNetPlans.json`).
> Swin UNETR: 96³ patches, batch 1, matched preprocessing.
>
> **Observations.** (examples of what to look for)
> - Which model wins per structure? ET is small → HD95 punishes misses; ED is diffuse → Dice differs less.
> - Variance: does one model have heavier tails (a few catastrophic cases)?
> - Efficiency: params, GPU memory, inference s/volume (see `model_stats.json`).
>
> **Interpretation.** Why might a self-configured CNN beat (or lose to) a pretrained Transformer
> at this data scale? (small-data regime, augmentation strength, inductive bias, pretraining domain gap)
>
> **Limitations.** subset size, reduced epochs, budget-matched ≠ tuning-matched, single fold,
> no ensembling/TTA for saliency runs, no significance test at n={N_val}.
"""),
])

# ============================================================================
# 05 — XAI analysis
# ============================================================================
nb(f"{ROOT}/05_xai_analysis.ipynb", [
md(f"""# 🔍 XAI — explaining 3D segmentation predictions (both arms)

**📎 Required Kaggle Datasets:**
1. Output of **notebook 03** (Swin UNETR checkpoint `swinunetr_best.pt` + split + results)
2. *(Optional)* snapshot from **notebook 02** (`nnunet_snapshots/...`) for nnU-Net saliency
3. BraTS dataset for the images

**What we do**
1. Gradient-based **input saliency** for the Swin UNETR (repo's `src/xai.py`) + MONAI **GradCAM**
2. *(optional cell)* saliency through **nnU-Net's own predictor** (single pass, TTA off — documented)
3. Side-by-side overlays on the **best-Dice and worst-Dice validation cases** (failure analysis)
4. Honest method notes: what gradient saliency can and cannot tell us

> ⚠️ Method transparency (put this in your writeup): input-gradient saliency is a basic
> attribution method — fast and model-agnostic, but noisy. GradCAM is smoother (it aggregates
> gradients at feature maps). We show both; where they disagree, we say so instead of
> cherry-picking the prettier map. This kind of honesty is exactly what XAI reviewers look for.
"""),
code(f"""!pip install -q -U "monai[nibabel]>=1.3"
!git clone --depth 1 https://github.com/{REPO_PLACEHOLDER}.git /kaggle/working/repo
import os, sys, glob, json
sys.path.insert(0, "/kaggle/working/repo"); os.chdir("/kaggle/working/repo")
import numpy as np, pandas as pd, torch, nibabel as nib
from src.xai import saliency_torch_model, plot_overlay"""),
md("## 1 · Load the trained Swin UNETR + pick analysis cases (best & worst Dice)"),
code("""from monai.networks.nets import SwinUNETR
device = torch.device("cuda")
model = SwinUNETR(img_size=(96,96,96), in_channels=4, out_channels=4, feature_size=24, use_v2=True).to(device)
ckpt_path = (glob.glob("/kaggle/input/**/swinunetr_best.pt", recursive=True) + glob.glob("/kaggle/working/swinunetr_best.pt"))[0]
ckpt = torch.load(ckpt_path, map_location=device); model.load_state_dict(ckpt["model"]); model.eval()
print("loaded", ckpt_path, "| val dice at save:", ckpt["val_dice"])

# split + per-case metrics
split = json.load(open((glob.glob("/kaggle/input/**/brats_subset_split.json", recursive=True) + glob.glob("/kaggle/working/**/brats_subset_split.json", recursive=True))[0]))
per_case = pd.read_csv((glob.glob("/kaggle/input/**/results/brats_swinunetr/per_case_metrics.csv", recursive=True) + glob.glob("/kaggle/working/results/brats_swinunetr/per_case_metrics.csv"))[0])
mean_dice = per_case.groupby("case")["dice"].mean().sort_values()
worst, best = mean_dice.index[0], mean_dice.index[-1]
print(f"worst case: {{worst}} (mean dice {{mean_dice.iloc[0]:.3f}}) | best: {{best}} ({{mean_dice.iloc[-1]:.3f}})")"""),
md("""## 2 · Rebuild the case loader (same preprocessing as training — critical!)

Saliency w.r.t. a *differently preprocessed* input is meaningless. We reuse the exact
normalization from notebook 03."""),
code("""MODS = ("flair", "t1", "t1ce", "t2")
_REMAP_LUT = np.array([0, 1, 2, 0, 3], dtype=np.int8)   # BraTS {0,1,2,4} -> {0,1,2,3}
cands = glob.glob("/kaggle/input/*brats*")
TD = (glob.glob(f"{cands[0]}/**/RSNA_ASNR_MICCAI_BraTS_2021_TrainingData", recursive=True) or [cands[0]])[0]
def load_case(cid):
    d = os.path.join(TD, split["id_map"][cid])
    X = np.stack([np.asarray(nib.load([f for f in glob.glob(os.path.join(d, f"*{m}.nii.gz")) if "seg" not in os.path.basename(f).lower()][0]).dataobj, dtype=np.float32) for m in MODS])
    y = _REMAP_LUT[np.clip(np.asarray(nib.load(glob.glob(os.path.join(d, "*seg.nii.gz"))[0]).dataobj).astype(np.int16), 0, 4)]
    return X, y
def normalize(x):
    fg = x[x > 0]
    if fg.size == 0: return x
    lo, hi = np.percentile(fg, [0.5, 99.5]); x = np.clip(x, lo, hi)
    fg = x[x > 0]
    return (x - fg.mean()) / (fg.std() + 1e-8)"""),
md("""## 3 · Saliency for one case, per tumor class (ET=3, ED=2, NCR=1)

The backward pass through the full 240×240×155 volume does not fit on a T4, so I explain a
**96³ window centred on the tumour** and say so in the figure title. The preprocessing inside
that window is identical to training — a saliency map computed on differently normalised
input would be meaningless."""),
code("""%%time
from monai.inferers import sliding_window_inference

CROP = 96

def tumor_crop(X, y, size=CROP):
    fg = np.argwhere(y > 0)
    c = fg.mean(0).astype(int) if len(fg) else np.array(X.shape[1:]) // 2
    starts = [int(np.clip(c[i] - size // 2, 0, X.shape[i + 1] - size)) for i in range(3)]
    sl = (slice(None),) + tuple(slice(s, s + size) for s in starts)
    return X[sl], y[starts[0]:starts[0]+size, starts[1]:starts[1]+size, starts[2]:starts[2]+size]

def analyze(cid, target_class, tag):
    X, y = load_case(cid)
    Xn = np.stack([normalize(c) for c in X]).astype(np.float32)
    Xc, yc = tumor_crop(Xn, y)
    xt = torch.from_numpy(Xc[None]).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(xt, (CROP, CROP, CROP), 1, model, overlap=0.5)
        pred = torch.argmax(torch.softmax(logits, 1), 1)[0].cpu().numpy()
    sal = saliency_torch_model(model, xt, target_class=target_class)   # (CROP, CROP, CROP)
    outp = f"/kaggle/working/figures/xai_{tag}_c{target_class}.png"
    plot_overlay(Xc[0], pred, yc.astype(np.uint8), sal, axis=2,
                 title=f"{cid} — class {target_class} — gradient saliency ({tag}, {CROP}³ crop)",
                 out_path=outp)
    return outp, pred, sal

class_names = {1: "NCR/NET", 2: "Edema", 3: "Enhancing tumor"}
for cid, tag in [(best, "best"), (worst, "worst")]:
    for c in (2, 3):
        p, pred, sal = analyze(cid, c, tag)
        print(f"{tag} case {cid} class {class_names[c]} -> {p}")"""),
md("""## 4 · MONAI GradCAM (smoother attribution — cross-check)

GradCAM aggregates gradients at a target feature layer instead of the raw input, so maps are
less noisy. Agreement between the two methods increases confidence in the explanation.""",),
code("""from monai.visualize import GradCAM
layer = "layers1"  # SwinUNETR encoder stage; try layers2/layers3 for coarser maps
try:
    cam = GradCAM(nn_module=model, target_layers=f"{layer}.0.blocks.0.attn")
    X, y = load_case(worst)
    Xn = np.stack([normalize(c) for c in X]).astype(np.float32)
    Xc, yc = tumor_crop(Xn, y)          # same 96^3 window as the gradient saliency
    xt = torch.from_numpy(Xc[None]).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(xt, (CROP, CROP, CROP), 1, model, overlap=0.5)
        pred = torch.argmax(torch.softmax(logits, 1), 1)[0].cpu().numpy()
    for c in (2, 3):
        res = cam(x=xt, class_idx=torch.tensor([c]).to(device))
        heat = res[0].mean(0).detach().cpu().numpy()
        if heat.shape != Xc.shape[1:]:   # GradCAM may return a coarser map
            heat = torch.nn.functional.interpolate(
                torch.from_numpy(heat)[None, None], size=Xc.shape[1:],
                mode="trilinear", align_corners=False)[0, 0].numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        plot_overlay(Xc[0], pred, yc.astype(np.uint8), heat, axis=2,
                     title=f"{worst} — class {c} — MONAI GradCAM ({layer}, {CROP}³ crop)",
                     out_path=f"/kaggle/working/figures/gradcam_worst_c{c}.png")
    print("✅ GradCAM figures saved")
except Exception as e:
    print("GradCAM target layer name differs in this MONAI version — introspect with:")
    print([n for n, _ in model.named_modules() if "blocks" in n][:12])
    print("error was:", e)"""),
md("""## 5 · (Optional) nnU-Net saliency via its own predictor

If notebook 02's **snapshot** (`nnunet_snapshots/Dataset701_BraTS21/...`) is attached, we can
compute saliency through nnU-Net's preprocessing + network — the repo's `nnunet_case_saliency`
handles the plumbing (single centered pass, TTA off, resampled back to original geometry).

> This cell needs the full nnU-Net env (env vars + preprocessed plans). If the snapshot lacks
> `plans.json`/dataset metadata, run notebook 02's setup cells first (no retraining needed).""",),
code("""# OPTIONAL — uncomment when the nnU-Net snapshot is attached
# !pip install -q -U nnunetv2
# import os
# os.environ["nnUNet_raw"]          = "/kaggle/working/nnunet_env/raw"
# os.environ["nnUNet_preprocessed"] = "/kaggle/working/nnunet_env/pre"
# os.environ["nnUNet_results"]      = "/kaggle/working/nnunet_env/res"
# # restore snapshot into $nnUNet_results, re-run plan_and_preprocess for Dataset701 if needed, then:
# from src.xai import nnunet_case_saliency, plot_overlay
# model_folder = "/kaggle/working/nnunet_env/res/Dataset701_BraTS21/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/fold_0"
# cid = worst
# d = os.path.join(TD, split["id_map"][cid])
# files = [[f for f in glob.glob(os.path.join(d, f"*{m}.nii.gz")) if "seg" not in os.path.basename(f).lower()][0] for m in MODS]
# out = nnunet_case_saliency(model_folder, files, target_class=3)
# img0 = np.asarray(nib.load(files[0]).dataobj, dtype=np.float32)
# gt = _REMAP_LUT[np.clip(np.asarray(nib.load(glob.glob(os.path.join(d, "*seg.nii.gz"))[0]).dataobj).astype(np.int16), 0, 4)]
# plot_overlay(img0, out["prediction"], gt, out["saliency_orig"], axis=2,
#              title=f"nnU-Net saliency — {cid} — ET", out_path=f"/kaggle/working/figures/nnunet_saliency_{cid}.png")"""),
md("""## 6 · Writing up the XAI analysis (the part professors actually read)

For each figure, answer in 2–3 sentences:
1. **Does the model look where a radiologist would?** (saliency on tumor vs healthy tissue)
2. **Failure modes:** on the worst case, is saliency diffuse (uncertain), misplaced (confusion
   with vessels/artifacts), or correct-but-undersegmented (threshold/sensitivity issue)?
3. **Method caveat:** gradient saliency is a first-order explanation — noisy, no guarantees.
   State it. If maps disagree between methods, report the disagreement.

Suggested repo layout for results:
```
docs/figures/xai_best_c3.png
docs/figures/xai_worst_c2.png
docs/figures/gradcam_worst_c3.png
```
"""),
])

print("\nAll notebooks generated ✅")
