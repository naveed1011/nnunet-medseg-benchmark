"""Tests for the dataset generators/converters (no download, all synthetic)."""

from __future__ import annotations

import json

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from scripts.make_demo_data import _write_dataset
from data.convert_to_nnunet import convert_brats, convert_heart


# --------------------------------------------------------------------------- #
# synthetic phantom (used by the smoke test and by CI)
# --------------------------------------------------------------------------- #

def test_demo_dataset_is_valid_nnunet_format(tmp_path):
    ds = _write_dataset(tmp_path, n_cases=5, shape=(24, 24, 16),
                        spacing=(1.0, 1.0, 1.5), n_val=2, seed=0, force=True)

    meta = json.loads((ds / "dataset.json").read_text())
    assert meta["numTraining"] == 5
    assert meta["file_ending"] == ".nii.gz"
    assert set(meta["channel_names"]) == {"0", "1"}
    assert sorted(meta["labels"].values()) == [0, 1, 2, 3]
    # nnU-Net requires consecutive labels starting at 0
    assert list(meta["labels"].values()) == list(range(4))

    images = sorted((ds / "imagesTr").glob("*.nii.gz"))
    labels = sorted((ds / "labelsTr").glob("*.nii.gz"))
    assert len(images) == 5 * 2
    assert len(labels) == 5

    split = json.loads((ds / "demo_split.json").read_text())
    assert len(split["val"]) == 2
    assert set(split["train"]).isdisjoint(split["val"])
    assert sorted(split["train"] + split["val"]) == sorted(
        p.name[: -len(".nii.gz")] for p in labels)


def test_demo_labels_are_within_declared_range(tmp_path):
    ds = _write_dataset(tmp_path, n_cases=3, shape=(24, 24, 16),
                        spacing=(1.0, 1.0, 1.5), n_val=1, seed=3, force=True)
    for f in (ds / "labelsTr").glob("*.nii.gz"):
        arr = np.asarray(nib.load(str(f)).dataobj)
        assert set(np.unique(arr).tolist()) <= {0, 1, 2, 3}
        assert arr.dtype == np.uint8


def test_demo_is_deterministic(tmp_path):
    a = _write_dataset(tmp_path / "a", 3, (24, 24, 16), (1.0, 1.0, 1.5), 1, seed=7, force=True)
    b = _write_dataset(tmp_path / "b", 3, (24, 24, 16), (1.0, 1.0, 1.5), 1, seed=7, force=True)
    va = np.asarray(nib.load(str(a / "imagesTr" / "phantom_000_0000.nii.gz")).dataobj)
    vb = np.asarray(nib.load(str(b / "imagesTr" / "phantom_000_0000.nii.gz")).dataobj)
    assert np.array_equal(va, vb)


def test_demo_refuses_to_overwrite_without_force(tmp_path):
    _write_dataset(tmp_path, 2, (16, 16, 16), (1.0, 1.0, 1.0), 1, seed=0, force=True)
    with pytest.raises(SystemExit, match="already exists"):
        _write_dataset(tmp_path, 2, (16, 16, 16), (1.0, 1.0, 1.0), 1, seed=0, force=False)


# --------------------------------------------------------------------------- #
# MSD Heart converter
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_msd_heart(tmp_path):
    raw = tmp_path / "Task02_Heart"
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        case = f"la_{i:03d}"
        img = rng.random((16, 16, 12)).astype(np.float32)
        lab = (rng.random((16, 16, 12)) > 0.7).astype(np.uint8)
        nib.save(nib.Nifti1Image(img, np.eye(4)), str(raw / "imagesTr" / f"{case}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(lab, np.eye(4)), str(raw / "labelsTr" / f"{case}.nii.gz"))
    return raw


def test_convert_heart(fake_msd_heart, tmp_path):
    out = convert_heart(fake_msd_heart, tmp_path / "raw", symlink=False)
    assert out.name == "Dataset002_Heart"
    meta = json.loads((out / "dataset.json").read_text())
    assert meta["numTraining"] == 3
    assert set(meta["channel_names"]) == {"0"}
    assert meta["labels"] == {"background": 0, "left atrium": 1}
    assert len(list((out / "imagesTr").glob("*_0000.nii.gz"))) == 3
    assert len(list((out / "labelsTr").glob("*.nii.gz"))) == 3


def test_convert_heart_rejects_wrong_layout(tmp_path):
    bad = tmp_path / "not_msd"
    bad.mkdir()
    with pytest.raises(SystemExit, match="does not look like an MSD task"):
        convert_heart(bad, tmp_path / "raw", symlink=False)


# --------------------------------------------------------------------------- #
# BraTS converter (label remap 4 -> 3 is the easy thing to get wrong)
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_brats(tmp_path):
    root = tmp_path / "brats"
    for i in range(2):
        case = root / f"BraTS2021_{i:05d}"
        case.mkdir(parents=True)
        rng = np.random.default_rng(i)
        for mod in ("flair", "t1", "t1ce", "t2"):
            vol = rng.random((12, 12, 8)).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, np.eye(4)),
                     str(case / f"BraTS2021_{i:05d}_{mod}.nii.gz"))
        seg = np.zeros((12, 12, 8), dtype=np.int16)
        seg[2:4, 2:4, 2:4] = 1
        seg[2:5, 2:5, 2:5] = 2
        seg[3:5, 3:5, 3:5] = 4
        nib.save(nib.Nifti1Image(seg, np.eye(4)),
                 str(case / f"BraTS2021_{i:05d}_seg.nii.gz"))
    return root


def test_convert_brats_remaps_labels(fake_brats, tmp_path):
    out = convert_brats(fake_brats, tmp_path / "raw", symlink=False, max_cases=None)
    meta = json.loads((out / "dataset.json").read_text())
    assert meta["numTraining"] == 2
    assert set(meta["channel_names"]) == {"0", "1", "2", "3"}

    for f in (out / "labelsTr").glob("*.nii.gz"):
        values = set(np.unique(np.asarray(nib.load(str(f)).dataobj)).tolist())
        assert values <= {0, 1, 2, 3}
        assert 4 not in values


def test_convert_brats_max_cases(fake_brats, tmp_path):
    out = convert_brats(fake_brats, tmp_path / "raw", symlink=False, max_cases=1)
    assert json.loads((out / "dataset.json").read_text())["numTraining"] == 1
    assert len(list((out / "imagesTr").glob("*_0000.nii.gz"))) == 1
