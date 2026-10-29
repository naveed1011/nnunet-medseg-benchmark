"""Tests for saliency and the overlay figure (skipped when torch is absent)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.xai import plot_overlay, saliency_torch_model


class TinySegNet(torch.nn.Module):
    """A 2-class 3D segmentation toy: enough to get a real gradient."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_saliency_shape_and_normalisation():
    torch.manual_seed(0)
    model = TinySegNet()
    x = torch.randn(1, 1, 8, 8, 8)
    sal = saliency_torch_model(model, x, target_class=1)

    assert sal.shape == (8, 8, 8)
    assert np.isfinite(sal).all()
    assert sal.min() >= 0.0
    assert sal.max() == pytest.approx(1.0)
    assert (sal > 0).sum() > 0


def test_saliency_does_not_mutate_input():
    torch.manual_seed(0)
    model = TinySegNet()
    x = torch.randn(1, 1, 8, 8, 8)
    before = x.clone()
    saliency_torch_model(model, x, target_class=0)
    assert torch.equal(x, before)


def test_saliency_localises_a_bright_blob():
    """A conv with a bright input region should blame that region most."""
    torch.manual_seed(0)
    model = TinySegNet()
    x = torch.zeros(1, 1, 12, 12, 12)
    x[..., 2:6, 2:6, 2:6] = 5.0
    sal = saliency_torch_model(model, x, target_class=1)
    assert sal[2:6, 2:6, 2:6].mean() > sal[8:, 8:, 8:].mean()


def test_plot_overlay_all_panels(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.normal(size=(16, 16, 12)).astype(np.float32)
    pred = (rng.random((16, 16, 12)) > 0.6).astype(np.int16)
    gt = (rng.random((16, 16, 12)) > 0.6).astype(np.int16)
    sal = rng.random((16, 16, 12)).astype(np.float32)
    out = tmp_path / "overlay.png"
    plot_overlay(image, pred, gt, sal, title="test", out_path=out)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_overlay_image_only(tmp_path):
    image = np.random.default_rng(1).random((8, 8, 8)).astype(np.float32)
    out = tmp_path / "only.png"
    plot_overlay(image, out_path=out, title="image only")
    assert out.exists() and out.stat().st_size > 1000


def test_plot_overlay_shape_mismatch_raises():
    image = np.zeros((8, 8, 8), np.float32)
    with pytest.raises(ValueError, match="shape"):
        plot_overlay(image, pred=np.zeros((4, 4, 4)), gt=None, saliency=None)
