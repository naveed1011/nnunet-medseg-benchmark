"""
Convert raw datasets into the nnU-Net v2 dataset format.

nnU-Net v2 expects:

    nnUNet_raw/
    └── DatasetXXX_Name/
        ├── dataset.json          # channels, labels, number of training cases
        ├── imagesTr/             # case_0000.nii.gz, case_0001.nii.gz, ... (per channel)
        ├── labelsTr/             # case.nii.gz (integer labels)
        └── imagesTs/ (optional)

Supported tasks here:

    --task heart   MSD Task02_Heart        -> Dataset002_Heart   (1 channel, 1 FG class)
    --task brats   BraTS 2021 (Kaggle dir) -> Dataset701_BraTS21 (4 channels, 3 FG classes)

BraTS specifics handled for you:
  * 4 MRI modalities mapped to channel suffixes _0000.._0003 (flair, t1, t1ce, t2)
  * Original label values {0: background, 1: NCR/NET, 2: ED, 4: ET} are remapped to
    the CONSECUTIVE nnU-Net convention {0: background, 1: NCR/NET, 2: ED, 3: ET}
  * Optional --max-cases to build a fast Kaggle-friendly subset (default: all)

Usage
-----
    python data/convert_to_nnunet.py --task heart --raw data/raw/Task02_Heart
    python data/convert_to_nnunet.py --task brats --raw data/raw/brats2021 --max-cases 100
    python data/convert_to_nnunet.py --task brats --raw /kaggle/input/brats-2021-task-data --max-cases 50

By default writes into ./nnUNet_raw (override with --out). Remember that
nnUNet_raw / nnUNet_preprocessed / nnUNet_results locations are controlled by
the matching environment variables — keep them consistent!
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

MODALITIES = ("flair", "t1", "t1ce", "t2")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _out_root(user_out: str | None) -> Path:
    root = Path(user_out) if user_out else Path(os.environ.get("nnUNet_raw", "nnUNet_raw"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass  # fall back to copy (e.g. Windows without privileges)
    shutil.copy2(src, dst)


# --------------------------------------------------------------------------- #
# MSD Task02_Heart -> Dataset002_Heart
# --------------------------------------------------------------------------- #

def convert_heart(raw: Path, out_root: Path, symlink: bool) -> Path:
    images_tr = raw / "imagesTr"
    labels_tr = raw / "labelsTr"
    if not images_tr.is_dir() or not labels_tr.is_dir():
        raise SystemExit(
            f"{raw} does not look like an MSD task folder (missing imagesTr/labelsTr). "
            "Download it first: python data/download_msd.py --task Task02_Heart"
        )

    dst = out_root / "Dataset002_Heart"
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name[: -len("_0000.nii.gz")] for p in images_tr.glob("*_0000.nii.gz"))
    for case in cases:
        _link_or_copy(images_tr / f"{case}_0000.nii.gz", dst / "imagesTr" / f"{case}_0000.nii.gz", symlink)
        _link_or_copy(labels_tr / f"{case}.nii.gz", dst / "labelsTr" / f"{case}.nii.gz", symlink)

    dataset_json = {
        "name": "Dataset002_Heart",
        "description": "MSD Task02_Heart — left atrium segmentation from 3D cardiac MRI (LA)",
        "reference": "Medical Segmentation Decathlon (http://medicaldecathlon.com/), CC-BY-SA 4.0",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0",
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "channel_names": {"0": "MRI (late gadolinium enhancement 3D MRI)"},
        "labels": {"background": 0, "left atrium": 1},
    }
    (dst / "dataset.json").write_text(json.dumps(dataset_json, indent=4))
    print(f"✅ Dataset002_Heart ready at {dst} — {len(cases)} training cases, 1 channel, 1 FG class")
    return dst


# --------------------------------------------------------------------------- #
# BraTS 2021 -> Dataset701_BraTS21
# --------------------------------------------------------------------------- #

def convert_brats(raw: Path, out_root: Path, symlink: bool, max_cases: int | None) -> Path:
    import nibabel as nib

    # locate the training-data folder (handles Kaggle mirror nesting)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_brats import verify_layout  # reuse the same logic as the downloader

    train_dir = verify_layout(raw)
    case_dirs = sorted(p for p in train_dir.glob("BraTS2021_*") if p.is_dir())
    if not case_dirs:
        raise SystemExit(f"No BraTS2021_* case folders found under {train_dir}")
    if max_cases:
        case_dirs = case_dirs[:max_cases]
    print(f"🧠 Converting {len(case_dirs)} BraTS 2021 cases ...")

    dst = out_root / "Dataset701_BraTS21"
    (dst / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst / "labelsTr").mkdir(parents=True, exist_ok=True)

    remap = {0: 0, 1: 1, 2: 2, 4: 3}  # BraTS {NCR,ED,ET} -> consecutive {1,2,3}
    converted = []
    for i, cd in enumerate(case_dirs, 1):
        cid = cd.name
        mod_files = {}
        for m in MODALITIES:
            hits = sorted(cd.glob(f"*{m}.nii.gz"))
            hits = [h for h in hits if "seg" not in h.name.lower()]
            if not hits:
                print(f"   ⚠️  {cid}: modality '{m}' missing — skipping case")
                mod_files = None
                break
            mod_files[m] = hits[0]
        if mod_files is None:
            continue
        seg_hits = sorted(cd.glob("*seg.nii.gz"))
        if not seg_hits:
            print(f"   ⚠️  {cid}: no seg file — skipping case")
            continue

        new_id = f"BraTS21_{i:05d}"
        for ch, m in enumerate(MODALITIES):
            _link_or_copy(mod_files[m], dst / "imagesTr" / f"{new_id}_{ch:04d}.nii.gz", symlink)

        # remap labels (must rewrite the file — can't symlink)
        seg = nib.load(str(seg_hits[0]))
        arr = np.asarray(seg.dataobj).astype(np.uint8)
        lut = np.zeros(int(arr.max()) + 1, dtype=np.uint8)
        for old, new in remap.items():
            if old < len(lut):
                lut[old] = new
        unexpected = sorted(set(np.unique(arr).tolist()) - set(remap))
        if unexpected:
            print(f"   ⚠️  {cid}: unexpected label values {unexpected} — mapping to 0")
        arr = lut[arr]
        # fresh header: reusing the source header can carry scl_slope/scl_inter
        # that silently rescale integer labels on write
        out_img = nib.Nifti1Image(arr, seg.affine)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(dst / "labelsTr" / f"{new_id}.nii.gz"))

        converted.append(new_id)
        if i % 25 == 0 or i == len(case_dirs):
            print(f"   ... {i}/{len(case_dirs)} cases")

    dataset_json = {
        "name": "Dataset701_BraTS21",
        "description": "RSNA-ASNR-MICCAI BraTS 2021 — brain tumor segmentation (subset OK)",
        "reference": "Baid et al., arXiv:2107.02314; Kaggle mirror awsaf49/brats-2021-task-data",
        "licence": "see BraTS 2021 competition terms",
        "release": "1.0",
        "numTraining": len(converted),
        "file_ending": ".nii.gz",
        "channel_names": {"0": "FLAIR", "1": "T1", "2": "T1ce", "3": "T2"},
        "labels": {
            "background": 0,
            "necrotic and non-enhancing tumor (NCR/NET)": 1,
            "peritumoral edema (ED)": 2,
            "gadolinium-enhancing tumor (ET)": 3,
        },
    }
    (dst / "dataset.json").write_text(json.dumps(dataset_json, indent=4))
    print(f"✅ Dataset701_BraTS21 ready at {dst} — {len(converted)} cases, 4 channels, 3 FG classes")
    return dst


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["heart", "brats"])
    p.add_argument("--raw", required=True, help="Path to the raw dataset folder")
    p.add_argument("--out", default=None, help="nnUNet_raw root (default: $nnUNet_raw or ./nnUNet_raw)")
    p.add_argument("--max-cases", type=int, default=None, help="BraTS only: convert a subset (e.g. 100)")
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlinking (slower, safer across drives)")
    args = p.parse_args()

    raw = Path(args.raw).expanduser().resolve()
    if not raw.exists():
        raise SystemExit(f"Raw data path does not exist: {raw}")

    out_root = _out_root(args.out)
    if args.task == "heart":
        convert_heart(raw, out_root, symlink=not args.copy)
    else:
        convert_brats(raw, out_root, symlink=not args.copy, max_cases=args.max_cases)
