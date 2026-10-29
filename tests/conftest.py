"""Shared fixtures: tiny synthetic volumes written as real NIfTI files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

nib = pytest.importorskip("nibabel")


def make_volume(shape=(20, 20, 20), boxes=(), labels=(), dtype=np.int16):
    """Empty volume with axis-aligned boxes filled with a label.

    boxes and labels are parallel lists: (x0, x1, y0, y1, z0, z1) -> label.
    """
    vol = np.zeros(shape, dtype=dtype)
    for (x0, x1, y0, y1, z0, z1), label in zip(boxes, labels):
        vol[x0:x1, y0:y1, z0:z1] = label
    return vol


def save_nii(arr, path, spacing=(1.0, 1.0, 1.0)):
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nib.save(nib.Nifti1Image(np.asarray(arr), affine), str(path))


@pytest.fixture
def pred_gt_dirs(tmp_path):
    """Two cases, one label; case 'b' misses the structure entirely."""
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    gt_a = make_volume(boxes=[(4, 12, 4, 12, 4, 12)], labels=[1])
    pr_a = make_volume(boxes=[(4, 12, 4, 12, 4, 12)], labels=[1])   # perfect
    gt_b = make_volume(boxes=[(6, 14, 6, 14, 6, 14)], labels=[1])
    pr_b = make_volume()                                            # total miss

    save_nii(gt_a, gt_dir / "case_a.nii.gz")
    save_nii(pr_a, pred_dir / "case_a.nii.gz")
    save_nii(gt_b, gt_dir / "case_b.nii.gz")
    save_nii(pr_b, pred_dir / "case_b.nii.gz")
    return pred_dir, gt_dir


@pytest.fixture
def two_arm_csvs(tmp_path):
    """Per-case CSVs in the format src.metrics writes, for compare.py."""
    import pandas as pd

    rows = []
    rng = np.random.default_rng(0)
    for arm, offset in (("nnunet", 0.10), ("swinunetr", 0.0)):
        for case in range(6):
            for label in (1, 2, 3):
                rows.append({
                    "arm": arm,
                    "case": f"case_{case}",
                    "label": label,
                    "dice": float(np.clip(rng.normal(0.75 + offset - 0.05 * label, 0.08), 0, 1)),
                    "hd95_mm": float(abs(rng.normal(4.0, 1.5))),
                })
    df = pd.DataFrame(rows)
    out = {}
    for arm in ("nnunet", "swinunetr"):
        d = tmp_path / f"results_{arm}"
        d.mkdir()
        df[df["arm"] == arm].drop(columns="arm").to_csv(d / "per_case_metrics.csv", index=False)
        out[arm] = d
    return out
