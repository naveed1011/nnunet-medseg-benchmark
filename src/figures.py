"""
Figures that make a run inspectable at a glance.

These are intentionally dependency-light (matplotlib + numpy) and work on any
3D volume: they are used by the demo smoke test, by the tests, and by the
notebooks when something needs a quick qualitative check.

The two workhorses:

* `plot_prediction_case` — image | prediction | ground truth | error map for one
  slice. The error map is the honest panel: red = false positive, blue = missed.
* `plot_dataset_qc` — a grid of cases so you can see what the network is fed
  before blaming the architecture for a bad Dice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _display_plane(volume: np.ndarray, axis: int, index: int | None = None):
    """Return a 2D slice oriented the way we like to look at it."""
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    index = volume.shape[axis] // 2 if index is None else int(index)
    index = max(0, min(index, volume.shape[axis] - 1))
    sl = [slice(None)] * 3
    sl[axis] = index
    plane = volume[tuple(sl)]
    if axis != 2:
        plane = np.rot90(plane)
    return plane


def _image_limits(plane: np.ndarray) -> tuple[float, float]:
    finite = plane[np.isfinite(plane)]
    pos = finite[finite > 0]
    sample = pos if pos.size > 10 else finite
    if sample.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(sample, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def plot_prediction_case(
    image: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: str | Path,
    title: str = "",
    axis: int = 2,
    slice_idx: int | None = None,
    dpi: int = 150,
) -> Path:
    """4-panel figure: image, prediction, ground truth, and a false-positive /
    false-negative error map. This is the figure that tells you *how* a model is
    wrong, not just how wrong."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    for name, arr in (("image", image), ("pred", pred), ("gt", gt)):
        if np.shape(arr) != np.shape(image):
            raise ValueError(f"{name} shape {np.shape(arr)} != image shape {np.shape(image)}")

    img = _display_plane(np.asarray(image, dtype=np.float32), axis, slice_idx)
    pr = _display_plane(np.asarray(pred), axis, slice_idx)
    gt_s = _display_plane(np.asarray(gt), axis, slice_idx)
    vmin, vmax = _image_limits(img)

    n_labels = int(max(np.max(pr), np.max(gt_s), 1))
    # label 0 is background: make it transparent instead of the first tab10 colour
    from matplotlib.colors import ListedColormap
    _tab = plt.get_cmap("tab10")
    cmap = ListedColormap([(0, 0, 0, 0)] + [_tab(i) for i in range(1, n_labels + 1)])

    tp = (pr > 0) & (gt_s > 0)
    fp = (pr > 0) & (gt_s == 0)
    fn = (pr == 0) & (gt_s > 0)
    err = np.zeros(pr.shape, dtype=np.uint8)  # 0 background, 1 FP, 2 FN
    err[fp] = 1
    err[fn] = 2
    err_cmap = ListedColormap(["#00000000", "#E4572E", "#2E86AB"])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    axes[0].imshow(img.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("image")
    axes[1].imshow(pr.T, cmap=cmap, origin="lower", vmin=0, vmax=max(n_labels, 1),
                   interpolation="nearest")
    axes[1].set_title("prediction")
    axes[2].imshow(gt_s.T, cmap=cmap, origin="lower", vmin=0, vmax=max(n_labels, 1),
                   interpolation="nearest")
    axes[2].set_title("ground truth")
    axes[3].imshow(img.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[3].imshow(err.T, cmap=err_cmap, origin="lower", vmin=0, vmax=2, interpolation="nearest",
                   alpha=0.55)
    axes[3].set_title(f"error  (FP {int(fp.sum())} / FN {int(fn.sum())} px)")
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("black")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Saved {out_path}")
    return out_path


def plot_dataset_qc(
    images: list[np.ndarray],
    labels: list[np.ndarray],
    out_path: str | Path,
    titles: list[str] | None = None,
    axis: int = 2,
    slice_idx: int | None = None,
    dpi: int = 150,
) -> Path:
    """Grid of cases: image on the left, label map on the right, per row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not images:
        raise ValueError("need at least one case")
    titles = titles or [f"case {i}" for i in range(len(images))]
    rows = len(images)
    from matplotlib.colors import ListedColormap
    _tab = plt.get_cmap("tab10")
    n_labels = max(max(int(np.max(np.asarray(lab))), 1) for lab in labels)
    cmap = ListedColormap([(0, 0, 0, 0)] + [_tab(i) for i in range(1, n_labels + 1)])

    fig, axes = plt.subplots(rows, 2, figsize=(7.5, 3.4 * rows), squeeze=False)
    for r, (img3, lab3) in enumerate(zip(images, labels)):
        img = _display_plane(np.asarray(img3, dtype=np.float32), axis, slice_idx)
        lab = _display_plane(np.asarray(lab3), axis, slice_idx)
        vmin, vmax = _image_limits(img)
        axes[r][0].imshow(img.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[r][0].set_title(f"{titles[r]} — image")
        axes[r][1].imshow(img.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[r][1].imshow(lab.T, cmap=cmap, origin="lower", vmin=0,
                          vmax=n_labels, interpolation="nearest", alpha=0.6)
        axes[r][1].set_title(f"{titles[r]} — labels {sorted(np.unique(lab).tolist())}")
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle("Dataset QC — same slice, image vs label", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼️  Saved {out_path}")
    return out_path
