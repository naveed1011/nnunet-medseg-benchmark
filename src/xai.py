"""
XAI for 3D medical segmentation — gradient saliency & visualization.

Two entry points:

1. `saliency_torch_model(model, x, target_class)` — input-gradient saliency for
   ANY torch model (used for the MONAI Swin UNETR arm; MONAI's `GradCAM` is a
   good cross-check — see notebooks/05_xai_analysis.ipynb).

2. `nnunet_case_saliency(...)` — saliency for nnU-Net's own predictor:
   preprocesses one case exactly like nnU-Net does, runs the network,
   back-propagates into the *preprocessed input volume*, then inverts nnU-Net's
   own transform (resample -> un-crop -> un-transpose) so the map lands on the
   original MRI. The prediction is produced with nnU-Net's own export function,
   so it is geometrically identical to what `nnUNetv2_predict` would write.

3. `plot_overlay(...)` — publication-style mid-slice figures with
   image + prediction + GT + saliency heatmap.

Design notes (be transparent about these in your writeup):
  * Input-gradient saliency is a sanity-check-level attribution method. It is
    fast, framework-agnostic, and works for CNNs and Transformers alike, but it
    can be noisy. Grad-CAM (MONAI/grad-cam packages) aggregates gradients at
    feature maps and is smoother — use both and show whether they agree.
  * nnU-Net predicts with sliding-window + TTA (mirroring). For saliency we run
    a SINGLE forward pass on the preprocessed volume (no TTA, no sliding
    window) — document this! It keeps the gradient well-defined and is standard
    practice for XAI on segmentation networks.
  * Saliency is computed at the preprocessed resolution and resampled back with
    linear interpolation; it is a coarse map, not a voxel-exact claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# 1) Generic torch saliency (Swin UNETR / MONAI arm)
# --------------------------------------------------------------------------- #

def saliency_torch_model(model, x, target_class: int = 1):
    """Input-gradient saliency for an arbitrary torch segmentation model.

    Args:
        model: torch.nn.Module producing logits of shape (B, C, *spatial)
        x: input tensor (B, C_in, *spatial), any device/dtype
        target_class: class whose score is attributed back to the input

    Returns:
        saliency map, numpy array shaped like x[0] but single-channel
        (max over input channels, abs of gradient), normalised to 0..1.
    """
    import torch

    model.eval()
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    if isinstance(logits, (tuple, list)):          # some MONAI nets return tuples
        logits = logits[0]
    if logits.ndim == len(x.shape):
        score = logits[0, target_class].sum()      # sum-over-voxels score
    else:                                          # already batch-squeezed
        score = logits[target_class].sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    grad = x.grad[0].abs().max(dim=0).values       # (D, H, W) or (H, W)
    g = grad.detach().cpu().float().numpy()
    gmax = g.max()
    return g / gmax if gmax > 0 else g


# --------------------------------------------------------------------------- #
# 2) nnU-Net saliency (uses nnU-Net's own preprocessor + predictor)
# --------------------------------------------------------------------------- #

def nnunet_case_saliency(
    model_folder: str | Path,
    image_files: list[str | Path],
    target_class: int = 1,
    configuration: str | None = None,
    use_gpu: bool = True,
    fold: int = 0,
) -> dict:
    """Gradient saliency of one nnU-Net prediction w.r.t. the ORIGINAL image.

    Args:
        model_folder: e.g. nnUNet_results/Dataset701_BraTS21/
                      nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/fold_0
                      (the fold folder itself)
        image_files: modality files in channel order (case_0000.nii.gz, ...)
        target_class: segmentation class to explain (1..K)
        configuration: inferred from the folder name when None
        use_gpu: run on CUDA when available
        fold: which fold the checkpoint belongs to (default 0)

    Returns dict with keys:
        saliency_orig  : float32 array in ORIGINAL image geometry, 0..1
        prediction     : label map in original geometry (nnU-Net export path)
        affine, spacing, original_shape
    """
    import torch
    import nibabel as nib
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
        insert_crop_into_image,
    )
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    model_folder = Path(model_folder)
    if configuration is None:
        # .../nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0 -> 3d_fullres
        configuration = model_folder.parent.name.split("__")[-1]

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,                       # TTA off -> single differentiable pass
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(fold,),
        checkpoint_name=_find_checkpoint(model_folder),
    )
    assert predictor.configuration_manager is not None

    # --- preprocess exactly like nnU-Net does -------------------------------
    preprocessor = DefaultPreprocessor(verbose=False)
    data, _, props = preprocessor.run_case(
        [str(f) for f in image_files],
        None,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json,
    )
    data_t = torch.from_numpy(np.ascontiguousarray(data)).float().unsqueeze(0).to(device)
    data_t.requires_grad_(True)

    # --- forward + backward through the actual network ----------------------
    net = predictor.network
    net.eval()
    net.to(device)
    is_2d = len(predictor.configuration_manager.patch_size) == 2
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        if is_2d:
            # nnU-Net's 2d trainer feeds (B, C, H, W): run the slices as a batch
            # and put the logits back into (C, D, H, W) for the export step
            logits = net(data_t[0].permute(1, 0, 2, 3))     # (D, C, H, W)
            if isinstance(logits, (tuple, list)):          # deep supervision
                logits = logits[0]
            logits = logits.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, D, H, W)
        else:
            logits = net(data_t)
            if isinstance(logits, (tuple, list)):          # deep supervision -> full res
                logits = logits[0]
    logits = logits[0]                              # drop batch -> (C, D, H, W)

    score = logits[target_class].float().sum()
    net.zero_grad(set_to_none=True)
    score.backward()
    if data_t.grad is None:
        raise RuntimeError("Gradient did not reach the input — is the network in eval/inference mode?")
    sal_pre = data_t.grad[0].abs().max(dim=0).values.detach().float().cpu().numpy()

    # --- prediction through nnU-Net's own export (guaranteed geometry) -----
    pred_orig = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits.detach().cpu().float(),
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.plans_manager.get_label_manager(predictor.dataset_json),
        props,
    )

    # --- same inverse transform for the continuous saliency map ------------
    sal_orig = _saliency_to_original_space(sal_pre, props, predictor)
    smax = float(np.nanmax(sal_orig)) if np.size(sal_orig) else 0.0
    if smax > 0:
        sal_orig = sal_orig / smax

    img0 = nib.load(str(image_files[0]))
    return {
        "saliency_orig": np.asarray(sal_orig, dtype=np.float32),
        "prediction": np.asarray(pred_orig),
        "affine": img0.affine,
        "spacing": tuple(float(s) for s in img0.header.get_zooms()[:3]),
        "original_shape": tuple(int(s) for s in props["shape_before_cropping"]),
        "configuration": configuration,
        "target_class": target_class,
    }


def _saliency_to_original_space(sal_pre: np.ndarray, props: dict, predictor) -> np.ndarray:
    """Resample -> un-crop -> un-transpose, mirroring nnU-Net's export function."""
    from nnunetv2.inference.export_prediction import insert_crop_into_image

    pm = predictor.plans_manager
    cm = predictor.configuration_manager

    # spacing in the transposed frame (same expression nnU-Net uses)
    spacing_transposed = [props["spacing"][i] for i in pm.transpose_forward]
    if len(cm.spacing) == len(props["shape_after_cropping_and_before_resampling"]):
        current_spacing = list(cm.spacing)
    else:  # 2d configuration: slice spacing is untouched
        current_spacing = [spacing_transposed[0], *cm.spacing]

    resampled = cm.resampling_fn_probabilities(
        sal_pre[None].astype(np.float32),
        props["shape_after_cropping_and_before_resampling"],
        current_spacing,
        [props["spacing"][i] for i in pm.transpose_forward],
    )
    if hasattr(resampled, "cpu"):
        resampled = resampled.cpu().numpy()
    resampled = np.asarray(resampled)
    if resampled.ndim == 4:
        resampled = resampled[0]

    full = np.zeros(props["shape_before_cropping"], dtype=np.float32)
    full = insert_crop_into_image(full, resampled.astype(np.float32),
                                  props["bbox_used_for_cropping"])
    return np.asarray(full).transpose(pm.transpose_backward)


def _find_checkpoint(model_folder: str | Path) -> str:
    """Newer nnU-Net writes .safetensors, older .pth — pick whichever exists."""
    names = ("checkpoint_final.safetensors", "checkpoint_final.pth",
             "checkpoint_best.safetensors", "checkpoint_best.pth")
    for name in names:
        if (Path(model_folder) / name).is_file():
            return name
    raise FileNotFoundError(f"No checkpoint found in {model_folder} (looked for {names})")


# --------------------------------------------------------------------------- #
# 3) Visualization
# --------------------------------------------------------------------------- #

def plot_overlay(image, pred=None, gt=None, saliency=None, axis: int = 2,
                 slice_idx: int | None = None, title: str = "",
                 out_path: str | Path | None = None, show: bool = False):
    """Slice figure: MRI | prediction | GT | saliency.

    Any of pred/gt/saliency may be None; the figure then shows fewer panels.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {image.shape}")

    slice_idx = image.shape[axis] // 2 if slice_idx is None else int(slice_idx)
    slice_idx = max(0, min(slice_idx, image.shape[axis] - 1))
    sl = [slice(None)] * 3
    sl[axis] = slice_idx
    sl = tuple(sl)

    def plane(vol):
        if vol is None:
            return None
        arr = np.asarray(vol)
        if arr.shape != image.shape:
            raise ValueError(f"volume shape {arr.shape} != image shape {image.shape}")
        arr = arr[sl]
        return np.rot90(arr) if axis != 2 else arr

    img_s, pred_s, gt_s, sal_s = plane(image), plane(pred), plane(gt), plane(saliency)
    panels = [(img_s, "image", "gray")]
    if pred_s is not None:
        panels.append((pred_s, "prediction", "tab10"))
    if gt_s is not None:
        panels.append((gt_s, "ground truth", "tab10"))
    if sal_s is not None:
        panels.append((sal_s, "gradient saliency", "jet"))

    pos = img_s[img_s > 0]
    if pos.size > 10:
        vmin, vmax = np.percentile(pos, [1, 99])
    else:
        vmin, vmax = float(np.nanmin(img_s)), float(np.nanmax(img_s))
    if vmax <= vmin:
        vmax = vmin + 1.0

    n_labels = 1
    for lab in (pred_s, gt_s):
        if lab is not None:
            n_labels = max(n_labels, int(np.max(lab)))
    from matplotlib.colors import ListedColormap
    _tab = plt.get_cmap("tab10")
    cmap_labels = ListedColormap([(0, 0, 0, 0)] + [_tab(i) for i in range(1, n_labels + 1)])

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.3), squeeze=False)
    axes = axes[0]
    for ax, (arr, name, cmap) in zip(axes, panels):
        if name == "image":
            ax.imshow(arr.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        elif name == "gradient saliency":
            ax.imshow(img_s.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            ax.imshow(arr.T, cmap="jet", origin="lower", alpha=0.5, vmin=0, vmax=1)
        else:
            ax.imshow(arr.T, cmap=cmap_labels, origin="lower", vmin=0,
                      vmax=max(n_labels, 1), interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
        ax.set_facecolor("black")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"🖼️  Saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)
