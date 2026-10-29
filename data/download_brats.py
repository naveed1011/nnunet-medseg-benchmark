"""
Get the RSNA-ASNR-MICCAI BraTS 2021 dataset.

Two options (this script supports both):

1. Kaggle API download (recommended off-Kaggle):
       python data/download_brats.py --kaggle
   Requires `pip install kaggle` and your kaggle.json API token in ~/.kaggle/.
   Downloads `awsaf49/brats-2021-task-data` (~12 GB) into data/raw/brats2021.

2. You already have the data (e.g. it's attached as a Kaggle Dataset inside a
   notebook at /kaggle/input/brats-2021-task-data): just point the converter
   at that folder — no download needed:
       python data/convert_to_nnunet.py --task brats --raw /kaggle/input/brats-2021-task-data

Expected folder layout after download (Kaggle mirror of the official data):

    brats2021/
    ├── RSNA_ASNR_MICCAI_BraTS_2021_TrainingData/
    │   ├── BraTS2021_00000/
    │   │   ├── BraTS2021_00000_flair.nii.gz
    │   │   ├── BraTS2021_00000_t1.nii.gz
    │   │   ├── BraTS2021_00000_t1ce.nii.gz
    │   │   ├── BraTS2021_00000_t2.nii.gz
    │   │   └── BraTS2021_00000_seg.nii.gz
    │   └── ...
    └── name_mapping.csv / other metadata files
"""

import argparse
import shutil
import zipfile
from pathlib import Path

KAGGLE_SLUG = "awsaf49/brats-2021-task-data"


def download_via_kaggle(out_dir: Path) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise SystemExit(
            "kaggle package not found. Install it (`pip install kaggle`) and place your "
            "kaggle.json token in ~/.kaggle/ (get it from kaggle.com -> Account -> API)."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"⬇️  Downloading {KAGGLE_SLUG} via Kaggle API into {out_dir} ...")
    api.dataset_download_files(KAGGLE_SLUG, path=str(out_dir), quiet=False)

    zips = sorted(out_dir.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No zip found in {out_dir} — download may have failed.")
    for z in zips:
        print(f"📦 Extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out_dir)
        z.unlink()

    print(f"✅ Done: {out_dir}")
    return out_dir


def verify_layout(root: Path) -> Path:
    """Locate the training-data folder inside an arbitrary BraTS2021 root."""
    candidates = [
        root / "RSNA_ASNR_MICCAI_BraTS_2021_TrainingData",
        root / "BraTS2021_TrainingData",
        root,
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("BraTS2021_*")):
            return c
    # Kaggle mirrors sometimes nest one level deeper — search for it
    found = sorted(root.glob("**/RSNA_ASNR_MICCAI_BraTS_2021_TrainingData"))
    if found:
        return found[0]
    raise SystemExit(
        f"Could not find BraTS2021 training data under {root}. "
        "Expected a folder containing BraTS2021_XXXXX case directories."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kaggle", action="store_true", help="Download via Kaggle API")
    p.add_argument("--out", default="data/raw/brats2021", help="Output directory")
    p.add_argument("--verify", default=None, help="Only verify an existing BraTS root (path)")
    args = p.parse_args()

    if args.verify:
        td = verify_layout(Path(args.verify))
        n = len(list(td.glob("BraTS2021_*")))
        print(f"✅ Found training data: {td} ({n} case folders)")
    elif args.kaggle:
        out = Path(args.out)
        download_via_kaggle(out)
        td = verify_layout(out)
        n = len(list(td.glob("BraTS2021_*")))
        print(f"✅ Verified: {td} ({n} case folders)")
    else:
        p.print_help()
