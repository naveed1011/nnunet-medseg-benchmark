"""
Segmentation metrics for the nnU-Net vs Swin UNETR comparison study.

Computes per-case, per-label:
  * Dice Similarity Coefficient (DSC)
  * Hausdorff distance HD95 (in voxels AND mm using the image spacing)
  * Volume (voxels + mm^3) and volume error
  * Surface-relevant extras: precision / recall

Everything is deliberately dependency-light (numpy / scipy / nibabel) so it
runs identically on Kaggle and locally, and independent of nnU-Net internals —
you can point it at ANY folder of predicted .nii.gz labels vs ground truth.

CLI usage
---------
    python -m src.metrics \
        --pred predictions/Dataset002_3d_fullres_fold0 \
        --gt   nnUNet_raw/Dataset002_Heart/labelsTr \
        --out  results/heart_nnunet

Python usage
------------
    from src.metrics import evaluate_folders
    report = evaluate_folders(pred_dir, gt_dir, labels=(1,))
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure


def _nanmean(values) -> float:
    """Mean that returns NaN instead of a RuntimeWarning for an all-NaN slice."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(arr))


# --------------------------------------------------------------------------- #
# core metric functions
# --------------------------------------------------------------------------- #

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice for one binary mask. Returns nan if both masks are empty
    (undefined — we report it honestly instead of pretending it's 1.0)."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return float("nan")
    inter = np.logical_and(p, g).sum()
    return float(2.0 * inter / (p.sum() + g.sum()))


def hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """95th-percentile symmetric Hausdorff distance in mm. Returns inf when one
    mask is empty and the other is not (worst case), nan when both are empty."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return float("nan")
    if not p.any() or not g.any():
        return float("inf")

    # border voxels of each mask (surface approximation)
    conn = generate_binary_structure(3, 1)
    p_surf = p ^ binary_erosion(p, conn)
    g_surf = g ^ binary_erosion(g, conn)
    if not p_surf.any():
        p_surf = p
    if not g_surf.any():
        g_surf = g

    dt_g = distance_transform_edt(~g, sampling=spacing)
    dt_p = distance_transform_edt(~p, sampling=spacing)
    d_pg = dt_g[p_surf]   # distance from pred surface -> gt mask
    d_gp = dt_p[g_surf]   # distance from gt surface -> pred mask
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


# --------------------------------------------------------------------------- #
# folder-level evaluation
# --------------------------------------------------------------------------- #

@dataclass
class CaseMetrics:
    case: str
    label: int
    dice: float
    hd95_mm: float
    pred_volume_mm3: float
    gt_volume_mm3: float
    volume_error_mm3: float
    precision: float
    recall: float


@dataclass
class EvalReport:
    pred_dir: str
    gt_dir: str
    labels: list[int]
    n_cases: int = 0
    mean: dict = field(default_factory=dict)
    per_case: list = field(default_factory=list)
    eval_seconds: float = 0.0

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        import csv
        import math

        def _py(o):
            """Make numpy/NaN values JSON-safe (NaN/inf -> None)."""
            if isinstance(o, dict):
                return {k: _py(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_py(v) for v in o]
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                o = float(o)
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return None
            return o

        fieldnames = ["case", "label", "dice", "hd95_mm", "pred_volume_mm3",
                      "gt_volume_mm3", "volume_error_mm3", "precision", "recall"]
        with open(out_dir / "per_case_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.per_case)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(_py({k: v for k, v in asdict(self).items() if k != "per_case"}),
                      f, indent=2, allow_nan=False)
        print(f"💾 Saved {out_dir/'per_case_metrics.csv'} and {out_dir/'summary.json'}")


def _match_stem(fname: str) -> str:
    """Strip _0000.._9999 channel suffixes and .nii.gz so pred/gt names align."""
    stem = fname
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        stem = parts[0]
    return stem


def evaluate_folders(
    pred_dir: str | Path,
    gt_dir: str | Path,
    labels: tuple[int, ...] | None = None,
    ignore_missing: bool = True,
) -> EvalReport:
    """Evaluate every prediction in `pred_dir` against the matching GT in `gt_dir`."""
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    t0 = time.time()

    gt_index = {_match_stem(p.name): p for p in gt_dir.glob("*.nii.gz")}
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise SystemExit(f"No predictions (*.nii.gz) found in {pred_dir}")

    report = EvalReport(str(pred_dir), str(gt_dir), list(labels or []))
    per_case: list[CaseMetrics] = []
    used_labels = set(labels) if labels else set()

    for pf in pred_files:
        stem = _match_stem(pf.name)
        if stem not in gt_index:
            if ignore_missing:
                print(f"⚠️  no GT for {pf.name} — skipped")
                continue
            raise SystemExit(f"No GT for {stem}")
        gf = gt_index[stem]

        pred_img = nib.load(str(pf))
        gt_img = nib.load(str(gf))
        pred_arr = np.asarray(pred_img.dataobj).astype(np.int16)
        gt_arr = np.asarray(gt_img.dataobj).astype(np.int16)
        if pred_arr.shape != gt_arr.shape:
            raise SystemExit(f"{stem}: shape mismatch pred {pred_arr.shape} vs gt {gt_arr.shape}")
        spacing = tuple(float(s) for s in gt_img.header.get_zooms()[:3])
        vox_mm3 = float(np.prod(spacing))

        # evaluate every foreground label either requested or present in GT/pred
        present = (set(np.unique(gt_arr).tolist()) | set(np.unique(pred_arr).tolist())) - {0}
        case_labels = sorted(used_labels | present)
        for lb in case_labels:
            used_labels.add(lb)
            p, g = pred_arr == lb, gt_arr == lb
            tp = float(np.logical_and(p, g).sum())
            per_case.append(CaseMetrics(
                case=stem, label=int(lb),
                dice=dice_score(p, g),
                hd95_mm=hd95_mm(p, g, spacing),
                pred_volume_mm3=float(p.sum()) * vox_mm3,
                gt_volume_mm3=float(g.sum()) * vox_mm3,
                volume_error_mm3=(float(p.sum()) - float(g.sum())) * vox_mm3,
                precision=tp / p.sum() if p.sum() else float("nan"),
                recall=tp / g.sum() if g.sum() else float("nan"),
            ))
        report.n_cases += 1

    report.per_case = [asdict(c) for c in per_case]
    report.labels = sorted(used_labels)

    # aggregate: mean per label (nanmean so empty-empty cases don't corrupt),
    # hd95 ignores inf but we also report %cases with inf (=missed structure)
    means: dict[str, dict[str, float]] = {}
    for lb in report.labels:
        rows = [c for c in per_case if c.label == lb]
        hd_finite = [c.hd95_mm for c in rows if np.isfinite(c.hd95_mm)]
        dice_vals = [c.dice for c in rows]
        means[f"label_{lb}"] = {
            "dice_mean": _nanmean(dice_vals),
            "dice_std": float(np.nanstd(dice_vals)) if dice_vals else float("nan"),
            "dice_median": float(np.nanmedian(dice_vals)) if dice_vals else float("nan"),
            "hd95_mm_mean": float(np.mean(hd_finite)) if hd_finite else float("nan"),
            "hd95_mm_p90": float(np.percentile(hd_finite, 90)) if hd_finite else float("nan"),
            "pct_cases_hd95_inf": float(100 * np.mean([not np.isfinite(c.hd95_mm) for c in rows])) if rows else float("nan"),
            "volume_error_mm3_mean": float(np.mean([c.volume_error_mm3 for c in rows])) if rows else float("nan"),
            "precision_mean": _nanmean([c.precision for c in rows]),
            "recall_mean": _nanmean([c.recall for c in rows]),
            "n_cases": float(len(rows)),
        }
    all_dice = [c.dice for c in per_case]
    means["overall"] = {
        "dice_mean": _nanmean(all_dice),
        "n_cases": float(report.n_cases),
    }
    report.mean = means
    report.eval_seconds = round(time.time() - t0, 2)
    return report


# --------------------------------------------------------------------------- #
# efficiency benchmark helper (params / memory / time) — used by compare.py
# --------------------------------------------------------------------------- #

def model_efficiency_stats(model, example_input_shape: tuple[int, ...] | None = None) -> dict:
    """Parameter count + (optionally) CUDA memory & forward-time for a torch model."""
    stats = {"n_params": sum(p.numel() for p in model.parameters())}
    try:
        import torch
        if example_input_shape is not None and torch.cuda.is_available():
            dev = torch.device("cuda")
            model = model.to(dev).eval()
            x = torch.randn(1, *example_input_shape, device=dev)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda"):
                _ = model(x)
            torch.cuda.synchronize()
            stats["forward_ms"] = round((time.time() - t0) * 1000, 1)
            stats["peak_cuda_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            model = model.cpu()
    except ImportError:
        pass
    return stats


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="Folder of predicted .nii.gz labels")
    p.add_argument("--gt", required=True, help="Folder of ground-truth .nii.gz labels")
    p.add_argument("--labels", type=int, nargs="*", default=None, help="Label IDs to evaluate (default: all found)")
    p.add_argument("--out", default="results/metrics", help="Output folder for CSV + summary.json")
    args = p.parse_args()

    report = evaluate_folders(args.pred, args.gt, tuple(args.labels) if args.labels else None)
    print("\n" + "=" * 70)
    print(f"Evaluated {report.n_cases} cases in {report.eval_seconds}s")
    for key, m in report.mean.items():
        print(f"  {key}: {json.dumps({k: round(v, 4) for k, v in m.items()})}")
    print("=" * 70)
    report.save(Path(args.out))


if __name__ == "__main__":
    main()
