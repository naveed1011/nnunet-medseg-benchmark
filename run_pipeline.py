#!/usr/bin/env python3
"""
One driver for the whole nnU-Net workflow — Windows, Linux and Kaggle.

The `scripts/*.sh` files are thin wrappers for people who live in bash; this
script is the portable path. It finds the nnU-Net entry points, sets the three
nnU-Net environment variables for every child process, and can run a complete
smoke test on synthetic data in one command.

Examples
--------
    # environment sanity check (no dataset needed)
    python run_pipeline.py doctor

    # full end-to-end smoke test on a synthetic phantom (CPU-friendly)
    python run_pipeline.py demo

    # the real thing, step by step
    python run_pipeline.py plan     --dataset 2
    python run_pipeline.py train    --dataset 2 --trainer nnUNetTrainer_100epochs
    python run_pipeline.py predict  --dataset 2 --input nnUNet_raw/Dataset002_Heart/imagesTr
    python run_pipeline.py evaluate --dataset 2
    python run_pipeline.py compare  --arm nnunet=results/heart --arm swin=results/heart_swin

Dataset IDs used by this repo: 2 = MSD Heart, 701 = BraTS subset, 900 = phantom.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_KEYS = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")


# --------------------------------------------------------------------------- #
# process / environment helpers
# --------------------------------------------------------------------------- #

def pipeline_env(extra: dict | None = None) -> dict:
    """Environment for nnU-Net children: the three required vars + sane CPU limits."""
    env = dict(os.environ)
    for key, folder in zip(ENV_KEYS, ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")):
        env.setdefault(key, str(ROOT / folder))
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    env.setdefault("nnUNet_n_proc_DA", "2")     # 12 default workers will thrash a laptop
    env.setdefault("nnUNet_compile", "False")   # torch.compile is a headache on Windows
    env.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4)))
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def find_tool(name: str) -> list[str]:
    """Locate an nnU-Net console script next to the current interpreter."""
    found = shutil.which(name)
    if found:
        return [found]
    exe = Path(sys.executable).parent / (name + (".exe" if os.name == "nt" else ""))
    if exe.exists():
        return [str(exe)]
    raise SystemExit(
        f"'{name}' not found. Install the framework first:\n"
        f"    {Path(sys.executable).name} -m pip install -U nnunetv2"
    )


def run(cmd: list[str], env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    env = env or pipeline_env()
    printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], env=env, cwd=str(ROOT))
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed with exit code {proc.returncode}: {printable}")
    print(f"  ↳ {time.time() - t0:.1f}s")
    return proc


def python(*args) -> list[str]:
    return [sys.executable, *map(str, args)]


def dataset_folder(dataset_id: int, root_key: str = "nnUNet_raw") -> Path:
    """Resolve DatasetXXX_* by numeric id (the folder name suffix varies)."""
    base = Path(pipeline_env()[root_key])
    hits = sorted(base.glob(f"Dataset{int(dataset_id):03d}_*"))
    if not hits:
        raise SystemExit(
            f"No Dataset{int(dataset_id):03d}_* folder in {base}.\n"
            f"Create it with data/convert_to_nnunet.py or scripts/make_demo_data.py."
        )
    return hits[0]


def inject_split(dataset_id: int, split: dict) -> Path:
    """Write fold 0 of splits_final.json so both model arms see identical data."""
    pre = dataset_folder(dataset_id, "nnUNet_preprocessed")
    path = pre / "splits_final.json"
    splits = [{"train": list(split["train"]), "val": list(split["val"])}]
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(f"🔧 Injected fold-0 split: {len(split['train'])} train / "
          f"{len(split['val'])} val -> {path}")
    return path


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_doctor(args) -> None:
    import importlib.metadata as md

    env = pipeline_env()
    print("== Python ==")
    print(f"  {sys.version.split()[0]}  ({sys.executable})")
    print("== Packages ==")
    for pkg in ("nnunetv2", "torch", "numpy", "nibabel", "scipy", "pandas", "matplotlib"):
        try:
            print(f"  {pkg:12s} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:12s} MISSING")
    try:
        import torch
        print(f"  torch CUDA available: {torch.cuda.is_available()} "
              f"(devices: {torch.cuda.device_count()})")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001 - doctor must never crash
        print(f"  torch check failed: {exc}")

    print("== nnU-Net environment ==")
    for key in ENV_KEYS:
        print(f"  {key:18s} {env[key]}")
    print("== Entry points ==")
    for tool in ("nnUNetv2_plan_and_preprocess", "nnUNetv2_train", "nnUNetv2_predict"):
        try:
            print(f"  {tool:32s} {find_tool(tool)[0]}")
        except SystemExit:
            print(f"  {tool:32s} MISSING")
    print("== Datasets ==")
    for key in ("nnUNet_raw", "nnUNet_preprocessed"):
        base = Path(env[key])
        found = sorted(p.name for p in base.glob("Dataset*")) if base.exists() else []
        print(f"  {key}: {found or '—'}")
    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    print(f"== Disk ==\n  {free:.1f} GB free in {ROOT}")


def cmd_plan(args) -> None:
    cmd = find_tool("nnUNetv2_plan_and_preprocess") + [
        "-d", str(args.dataset), "--verify_dataset_integrity", "-np", str(args.workers),
    ]
    if args.configs:
        cmd += ["-c", *args.configs]
    run(cmd)

    plans = dataset_folder(args.dataset, "nnUNet_preprocessed") / "nnUNetPlans.json"
    if plans.exists():
        data = json.loads(plans.read_text())
        for name, cfg in data.get("configurations", {}).items():
            if args.configs and name not in args.configs:
                continue
            print(f"\n📐 {name}: patch={cfg.get('patch_size')} batch={cfg.get('batch_size')} "
                  f"spacing={cfg.get('spacing')}")


def cmd_train(args) -> None:
    trainer = args.trainer
    if trainer is None:
        trainer = "nnUNetTrainer"
    cmd = find_tool("nnUNetv2_train") + [
        str(args.dataset), args.config, str(args.fold),
        "-tr", trainer, "--npz", "-device", args.device,
    ]
    if args.continue_training:
        cmd.append("--c")
    if args.val_only:
        cmd.append("--val")
    run(cmd)
    print(f"\n📈 Curves: {dataset_folder(args.dataset, 'nnUNet_results')}/"
          f"*{trainer}*/fold_{args.fold}/progress.png")


def _checkpoint_name(dataset_id: int, config: str, fold: str) -> str | None:
    """nnU-Net 2.5+ writes .safetensors; -chk needs the exact filename."""
    res = dataset_folder(dataset_id, "nnUNet_results")
    for folder in sorted(res.glob(f"*__{config}/fold_{fold}")):
        for name in ("checkpoint_final.safetensors", "checkpoint_final.pth"):
            if (folder / name).is_file():
                return name
    return None


def _predict(dataset_id: int, config: str, fold: str, input_dir: Path, out_dir: Path,
             disable_tta: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = find_tool("nnUNetv2_predict") + [
        "-i", input_dir, "-o", out_dir, "-d", str(dataset_id), "-c", config,
        "-f", str(fold),
    ]
    checkpoint = _checkpoint_name(dataset_id, config, fold)
    if checkpoint:
        cmd += ["-chk", checkpoint]
    if disable_tta:
        cmd.append("--disable_tta")
    run(cmd)
    return out_dir


def cmd_predict(args) -> None:
    input_dir = Path(args.input) if args.input else dataset_folder(args.dataset) / "imagesTr"
    out_dir = Path(args.out) if args.out else ROOT / "predictions" / f"dataset{args.dataset}_{args.config}_fold{args.fold}"
    _predict(args.dataset, args.config, args.fold, input_dir, out_dir, args.disable_tta)
    print(f"\n✅ {len(list(out_dir.glob('*.nii.gz')))} predictions in {out_dir}")


def cmd_evaluate(args) -> None:
    pred = Path(args.pred) if args.pred else ROOT / "predictions" / f"dataset{args.dataset}_{args.config}_fold{args.fold}"
    gt = Path(args.gt) if args.gt else dataset_folder(args.dataset) / "labelsTr"
    out = Path(args.out) if args.out else ROOT / "results" / f"dataset{args.dataset}_{args.config}_fold{args.fold}"
    cmd = python("-m", "src.metrics", "--pred", pred, "--gt", gt, "--out", out)
    if args.labels:
        cmd += ["--labels", *map(str, args.labels)]
    run(cmd)


def cmd_compare(args) -> None:
    if not args.arm:
        raise SystemExit("pass at least one --arm name=path")
    cmd = python("-m", "src.compare", "--out", args.out, "--task", args.task)
    for spec in args.arm:
        cmd += ["--arm", spec]
    run(cmd)


def _first_val_case(dataset_id: int) -> tuple[Path, Path, Path]:
    """Pick a validation case: (image files, pred path, gt path) from the fold split."""
    raw = dataset_folder(dataset_id)
    split_path = dataset_folder(dataset_id, "nnUNet_preprocessed") / "splits_final.json"
    split = json.loads(split_path.read_text())[0]
    case = split["val"][0]
    images = sorted(raw.glob(f"imagesTr/{case}_*.nii.gz"))
    if not images:
        images = sorted(raw.glob(f"imagesTr/{case}.nii.gz"))
    gt = raw / "labelsTr" / f"{case}.nii.gz"
    return images, gt, case


def cmd_figure(args) -> None:
    import nibabel as nib
    import numpy as np

    from src.figures import plot_prediction_case

    pred_dir = Path(args.pred)
    images, gt_path, case = _first_val_case(args.dataset)
    pred_path = pred_dir / f"{case}.nii.gz"
    if not pred_path.exists():
        raise SystemExit(f"No prediction for {case} in {pred_dir}")

    image = np.asarray(nib.load(str(images[0])).dataobj, dtype=np.float32)
    pred = np.asarray(nib.load(str(pred_path)).dataobj).astype(np.int16)
    gt = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.int16)
    plot_prediction_case(image, pred, gt, args.out,
                         title=f"Dataset{args.dataset} fold {args.fold} — case {case}",
                         axis=args.axis)


def cmd_xai(args) -> None:
    """Saliency through nnU-Net's own predictor (needs a trained model folder)."""
    from src.xai import nnunet_case_saliency, plot_overlay
    import nibabel as nib
    import numpy as np

    images, gt_path, case = _first_val_case(args.dataset)
    folder = args.model_folder
    if folder is None:
        res = dataset_folder(args.dataset, "nnUNet_results")
        hits = sorted(res.glob(f"*{args.config}*/fold_{args.fold}"))
        if not hits:
            raise SystemExit(f"No trained model under {res} for {args.config} fold {args.fold}")
        folder = hits[0]
    out = nnunet_case_saliency(folder, images, target_class=args.target_class,
                               configuration=args.config, use_gpu=False)
    image = np.asarray(nib.load(str(images[0])).dataobj, dtype=np.float32)
    gt = np.asarray(nib.load(str(gt_path)).dataobj)
    plot_overlay(image, out["prediction"], gt, out["saliency_orig"],
                 title=f"{case} — class {args.target_class} — nnU-Net gradient saliency",
                 out_path=args.out)


def cmd_demo(args) -> None:
    """Generate the phantom and walk the entire pipeline, including all figures."""
    env = pipeline_env()
    ds = 900
    cfg = args.config
    demo_root = ROOT / "demo"
    demo_root.mkdir(exist_ok=True)

    print("=" * 78)
    print("STEP 1/8 — synthetic phantom dataset")
    print("=" * 78)
    run(python(ROOT / "scripts" / "make_demo_data.py",
               "--out", env["nnUNet_raw"], "--n-cases", args.cases,
               "--force", "--qc-figure", demo_root / "figures" / "demo_data_qc.png"), env=env)

    print("\n" + "=" * 78)
    print(f"STEP 2/8 — plan & preprocess [{cfg}] (nnU-Net decides patch/batch/spacing itself)")
    print("=" * 78)
    run(find_tool("nnUNetv2_plan_and_preprocess") + [
        "-d", str(ds), "--verify_dataset_integrity", "-c", cfg, "-np", "2"], env=env)

    split = json.loads((dataset_folder(ds) / "demo_split.json").read_text())
    inject_split(ds, split)

    print("\n" + "=" * 78)
    print(f"STEP 3/8 — train fold 0 [{cfg}] (1 epoch)")
    print("=" * 78)
    run(find_tool("nnUNetv2_train") + [
        str(ds), cfg, "0", "-tr", "nnUNetTrainer_1epoch",
        "--npz", "-device", args.device], env=env)

    print("\n" + "=" * 78)
    print("STEP 4/8 — predict the held-out validation cases (with and without TTA)")
    print("=" * 78)
    val_in = demo_root / "val_images"
    val_in.mkdir(parents=True, exist_ok=True)
    for case in split["val"]:
        for f in sorted((dataset_folder(ds) / "imagesTr").glob(f"{case}_*.nii.gz")):
            shutil.copy2(f, val_in)
    tta = _predict(ds, cfg, "0", val_in, demo_root / "preds_tta")
    no_tta = _predict(ds, cfg, "0", val_in, demo_root / "preds_no_tta", disable_tta=True)

    print("\n" + "=" * 78)
    print("STEP 5/8 — evaluate with src.metrics (Dice + HD95 + volume error)")
    print("=" * 78)
    gt = dataset_folder(ds) / "labelsTr"
    for name, pred_dir in (("tta", tta), ("no_tta", no_tta)):
        run(python("-m", "src.metrics", "--pred", pred_dir, "--gt", gt,
                   "--labels", "1", "2", "3",
                   "--out", demo_root / "results" / name), env=env)

    print("\n" + "=" * 78)
    print("STEP 6/8 — qualitative figure (image | pred | GT | error map)")
    print("=" * 78)
    images, gt_path, case = _first_val_case(ds)
    import nibabel as nib
    import numpy as np

    from src.figures import plot_prediction_case
    plot_prediction_case(
        np.asarray(nib.load(str(images[0])).dataobj, dtype=np.float32),
        np.asarray(nib.load(str(tta / f"{case}.nii.gz")).dataobj).astype(np.int16),
        np.asarray(nib.load(str(gt_path)).dataobj).astype(np.int16),
        demo_root / "figures" / "demo_prediction.png",
        title=f"Phantom smoke test — {case} (held out) — nnU-Net 1 epoch",
    )

    print("\n" + "=" * 78)
    print("STEP 7/8 — XAI: gradient saliency through nnU-Net's own predictor")
    print("=" * 78)
    try:
        from src.xai import nnunet_case_saliency, plot_overlay
        folder = next(iter(sorted((dataset_folder(ds, "nnUNet_results")).glob(f"*__{cfg}/fold_0"))))
        payload = nnunet_case_saliency(folder, images, target_class=3,
                                       configuration=cfg, use_gpu=False)
        plot_overlay(np.asarray(nib.load(str(images[0])).dataobj, dtype=np.float32),
                     payload["prediction"],
                     np.asarray(nib.load(str(gt_path)).dataobj),
                     payload["saliency_orig"],
                     title=f"{case} — class 3 — nnU-Net gradient saliency",
                     out_path=demo_root / "figures" / "demo_saliency.png")
    except Exception as exc:  # noqa: BLE001 - XAI must not fail the smoke test
        print(f"⚠️  XAI step skipped: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 78)
    print("STEP 8/8 — comparison table + figures (TTA vs no-TTA: a real micro-ablation)")
    print("=" * 78)
    run(python("-m", "src.compare",
               "--arm", f"nnunet_tta={demo_root / 'results' / 'tta'}",
               "--arm", f"nnunet_no_tta={demo_root / 'results' / 'no_tta'}",
               "--out", demo_root / "comparison",
               "--task", "Synthetic phantom (smoke test)"), env=env)

    print("\n" + "=" * 78)
    print("SMOKE TEST COMPLETE")
    print("=" * 78)
    for p in sorted((demo_root / "figures").glob("*.png")):
        print(f"  figure: {p.relative_to(ROOT)}")
    for p in (demo_root / "comparison").glob("RESULTS.md"):
        print(f"  report: {p.relative_to(ROOT)}")
    print("\nThese numbers only prove the pipeline works — they are synthetic data.")
    print("Real results: run notebooks 01-05 on Kaggle (see TUTORIAL.md).")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check python/torch/nnU-Net/env/dataset setup").set_defaults(func=cmd_doctor)

    d = sub.add_parser("demo", help="full end-to-end smoke test on synthetic data")
    d.add_argument("--cases", type=int, default=12)
    d.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    d.add_argument("--config", default="2d", choices=["2d", "3d_fullres"],
                   help="2d is ~10x cheaper on CPU; use 3d_fullres on a GPU")
    d.set_defaults(func=cmd_demo)

    pl = sub.add_parser("plan", help="fingerprint + plan + preprocess")
    pl.add_argument("--dataset", type=int, required=True)
    pl.add_argument("--configs", nargs="*", default=None, help="e.g. 3d_fullres 2d")
    pl.add_argument("--workers", type=int, default=2)
    pl.set_defaults(func=cmd_plan)

    tr = sub.add_parser("train", help="train one configuration/fold")
    tr.add_argument("--dataset", type=int, required=True)
    tr.add_argument("--config", default="3d_fullres")
    tr.add_argument("--fold", default="0")
    tr.add_argument("--trainer", default=None, help="e.g. nnUNetTrainer_100epochs")
    tr.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    tr.add_argument("--continue", dest="continue_training", action="store_true")
    tr.add_argument("--val-only", action="store_true")
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="sliding-window inference")
    pr.add_argument("--dataset", type=int, required=True)
    pr.add_argument("--config", default="3d_fullres")
    pr.add_argument("--fold", default="0")
    pr.add_argument("--input", default=None)
    pr.add_argument("--out", default=None)
    pr.add_argument("--disable-tta", action="store_true")
    pr.set_defaults(func=cmd_predict)

    ev = sub.add_parser("evaluate", help="Dice/HD95 against ground truth")
    ev.add_argument("--dataset", type=int, required=True)
    ev.add_argument("--config", default="3d_fullres")
    ev.add_argument("--fold", default="0")
    ev.add_argument("--pred", default=None)
    ev.add_argument("--gt", default=None)
    ev.add_argument("--out", default=None)
    ev.add_argument("--labels", type=int, nargs="*", default=None)
    ev.set_defaults(func=cmd_evaluate)

    cp = sub.add_parser("compare", help="table + figures + RESULTS.md for N arms")
    cp.add_argument("--arm", action="append", help="name=path (repeat)")
    cp.add_argument("--out", default="docs/results/comparison")
    cp.add_argument("--task", default="3D medical segmentation")
    cp.set_defaults(func=cmd_compare)

    fg = sub.add_parser("figure", help="prediction-vs-GT figure for one val case")
    fg.add_argument("--dataset", type=int, required=True)
    fg.add_argument("--pred", required=True)
    fg.add_argument("--out", default="docs/figures/prediction.png")
    fg.add_argument("--fold", default=0)
    fg.add_argument("--axis", type=int, default=2)
    fg.set_defaults(func=cmd_figure)

    xa = sub.add_parser("xai", help="gradient saliency through nnU-Net")
    xa.add_argument("--dataset", type=int, required=True)
    xa.add_argument("--model-folder", default=None)
    xa.add_argument("--config", default="3d_fullres")
    xa.add_argument("--fold", default="0")
    xa.add_argument("--target-class", type=int, default=1)
    xa.add_argument("--out", default="docs/figures/saliency.png")
    xa.set_defaults(func=cmd_xai)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
