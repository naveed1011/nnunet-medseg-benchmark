"""
Download a Medical Segmentation Decathlon (MSD) task.

The MSD tasks are hosted on Google Drive, so we use `gdown` for the download.
Task02_Heart is the smallest task (20 cases, LA segmentation from 3D MRI) and
is the perfect "hello world" for learning the nnU-Net v2 pipeline.

Usage
-----
    python data/download_msd.py --task Task02_Heart
    python data/download_msd.py --task Task02_Heart --out data/raw   # custom folder

Notes
-----
- Files are ~2 GB for Task02_Heart; make sure you have disk space.
- On Kaggle you usually don't need this script — see docs/KAGGLE_GUIDE.md for
  how to attach the MSD as a Kaggle Dataset instead (faster, no download).
"""

import argparse
import tarfile
from pathlib import Path

# Official Google Drive IDs from http://medicaldecathlon.com/
MSD_GDRIVE_IDS = {
    "Task01_BrainTumour": "1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU",
    "Task02_Heart": "1wEB2I6S6tQBVEPxir8cB7kFB8gTQiQiT",
    "Task03_Liver": "1jyNUUGQ3re8dkF3KN7D7nOMVtxYrOa5",
    "Task04_Hippocampus": "1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C",
    "Task05_Prostate": "1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a",
    "Task06_Lung": "1I1LR7XjyEZ-VffQ-N4TL630iFK0AePv",
    "Task07_Pancreas": "1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL",
    "Task08_HepaticVessel": "1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS",
    "Task09_Spleen": "1jzeNU1EKnK81PyTsrxPVujfNlH0Ya1Ti",
    "Task10_Colon": "1m7tMpE9JegC1Sfc6h1TrIZmN-in0i461",
}


def download_task(task: str, out_dir: Path, keep_tar: bool = False) -> Path:
    try:
        import gdown
    except ImportError as e:
        raise SystemExit("gdown is required: pip install gdown") from e

    if task not in MSD_GDRIVE_IDS:
        raise SystemExit(f"Unknown task '{task}'. Choose from: {sorted(MSD_GDRIVE_IDS)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"{task}.tar"
    url = f"https://drive.google.com/uc?id={MSD_GDRIVE_IDS[task]}"

    print(f"⬇️  Downloading {task} from Google Drive ...")
    gdown.download(url, str(tar_path), quiet=False, fuzzy=True)

    print(f"📦 Extracting {tar_path.name} ...")
    with tarfile.open(tar_path) as tf:
        # `filter="data"` is the safe default on Python 3.12+; older Pythons
        # don't know the argument at all.
        try:
            tf.extractall(out_dir, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(out_dir)

    if not keep_tar:
        tar_path.unlink()

    task_dir = out_dir / task
    assert task_dir.is_dir(), f"Expected {task_dir} after extraction — check the download!"
    print(f"✅ Done: {task_dir}")
    return task_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Task02_Heart", help="MSD task name, e.g. Task02_Heart")
    p.add_argument("--out", default="data/raw", help="Output directory (default: data/raw)")
    p.add_argument("--keep-tar", action="store_true", help="Keep the downloaded .tar archive")
    args = p.parse_args()
    download_task(args.task, Path(args.out), keep_tar=args.keep_tar)
