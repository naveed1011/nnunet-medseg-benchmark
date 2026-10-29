"""Unit tests for the metric implementations and the folder-level evaluator."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.metrics import dice_score, evaluate_folders, hd95_mm


# --------------------------------------------------------------------------- #
# single-metric behaviour
# --------------------------------------------------------------------------- #

def test_dice_perfect_match_is_one():
    a = np.zeros((10, 10, 10), bool)
    a[2:6, 2:6, 2:6] = True
    assert dice_score(a, a) == pytest.approx(1.0)


def test_dice_disjoint_is_zero():
    a = np.zeros((10, 10, 10), bool)
    b = np.zeros((10, 10, 10), bool)
    a[0:3, 0:3, 0:3] = True
    b[6:9, 6:9, 6:9] = True
    assert dice_score(a, b) == pytest.approx(0.0)


def test_dice_half_overlap():
    a = np.zeros((10, 10, 10), bool)
    b = np.zeros((10, 10, 10), bool)
    a[0:4, 0:4, 0:4] = True          # 64 voxels
    b[2:6, 0:4, 0:4] = True          # 64 voxels, 32 overlap
    assert dice_score(a, b) == pytest.approx(2 * 32 / 128)


def test_dice_empty_empty_is_nan_not_one():
    empty = np.zeros((5, 5, 5), bool)
    assert np.isnan(dice_score(empty, empty))


def test_hd95_identical_masks_is_zero():
    a = np.zeros((20, 20, 20), bool)
    a[5:12, 5:12, 5:12] = True
    assert hd95_mm(a, a, (1.0, 1.0, 1.0)) == pytest.approx(0.0, abs=1e-6)


def test_hd95_two_voxel_shift_in_mm():
    gt = np.zeros((20, 20, 20), bool)
    gt[8:15, 6:12, 6:12] = True
    pred = np.zeros((20, 20, 20), bool)
    pred[6:13, 6:12, 6:12] = True     # shifted 2 voxels along axis 0
    assert hd95_mm(pred, gt, (1.0, 1.0, 1.0)) == pytest.approx(2.0, abs=1e-6)


def test_hd95_scales_with_spacing():
    gt = np.zeros((20, 20, 20), bool)
    gt[8:15, 6:12, 6:12] = True
    pred = np.zeros((20, 20, 20), bool)
    pred[6:13, 6:12, 6:12] = True
    assert hd95_mm(pred, gt, (2.0, 1.0, 1.0)) == pytest.approx(4.0, abs=1e-6)


def test_hd95_missed_structure_is_inf():
    gt = np.zeros((10, 10, 10), bool)
    gt[2:5, 2:5, 2:5] = True
    empty = np.zeros((10, 10, 10), bool)
    assert np.isinf(hd95_mm(empty, gt, (1.0, 1.0, 1.0)))
    assert np.isnan(hd95_mm(empty, empty, (1.0, 1.0, 1.0)))


# --------------------------------------------------------------------------- #
# folder evaluation
# --------------------------------------------------------------------------- #

def test_evaluate_folders_end_to_end(pred_gt_dirs, tmp_path):
    pred_dir, gt_dir = pred_gt_dirs
    report = evaluate_folders(pred_dir, gt_dir, labels=(1,))

    assert report.n_cases == 2
    rows = {(r["case"], r["label"]): r for r in report.per_case}
    assert rows[("case_a", 1)]["dice"] == pytest.approx(1.0)
    assert rows[("case_b", 1)]["dice"] == pytest.approx(0.0)
    # a total miss must be reported as inf, not silently dropped
    assert np.isinf(rows[("case_b", 1)]["hd95_mm"])

    mean = report.mean["label_1"]
    assert mean["dice_mean"] == pytest.approx(0.5)
    assert mean["pct_cases_hd95_inf"] == pytest.approx(50.0)
    assert mean["n_cases"] == 2

    out = tmp_path / "out"
    report.save(out)
    assert (out / "per_case_metrics.csv").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_cases"] == 2
    assert summary["mean"]["label_1"]["dice_mean"] == pytest.approx(0.5)


def test_evaluate_finds_unexpected_prediction_label(pred_gt_dirs, tmp_path):
    """A label the model invents must appear in the report (dice 0), not vanish."""
    pred_dir, gt_dir = pred_gt_dirs
    import nibabel as nib

    extra = np.zeros((20, 20, 20), dtype=np.int16)
    extra[1:3, 1:3, 1:3] = 7
    nib.save(nib.Nifti1Image(extra, np.eye(4)), str(pred_dir / "case_c.nii.gz"))
    gt = np.zeros((20, 20, 20), dtype=np.int16)
    gt[5:9, 5:9, 5:9] = 1
    nib.save(nib.Nifti1Image(gt, np.eye(4)), str(gt_dir / "case_c.nii.gz"))

    report = evaluate_folders(pred_dir, gt_dir, labels=(1,))
    assert 7 in report.labels
    row = next(r for r in report.per_case if r["label"] == 7)
    assert row["dice"] == pytest.approx(0.0)


def test_shape_mismatch_raises(pred_gt_dirs):
    pred_dir, gt_dir = pred_gt_dirs
    import nibabel as nib

    bad = np.zeros((10, 10, 10), dtype=np.int16)
    nib.save(nib.Nifti1Image(bad, np.eye(4)), str(pred_dir / "case_a.nii.gz"))
    with pytest.raises(SystemExit, match="shape mismatch"):
        evaluate_folders(pred_dir, gt_dir, labels=(1,))


def test_no_predictions_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    gt = tmp_path / "gt"
    gt.mkdir()
    with pytest.raises(SystemExit, match="No predictions"):
        evaluate_folders(empty, gt)


def test_channel_suffix_is_stripped(tmp_path):
    """`la_001_0000.nii.gz` must be matched against `la_001.nii.gz`."""
    from src.metrics import _match_stem

    assert _match_stem("la_001_0000.nii.gz") == "la_001"
    assert _match_stem("la_001.nii.gz") == "la_001"
    assert _match_stem("BraTS21_00001_seg.nii.gz") == "BraTS21_00001_seg"
