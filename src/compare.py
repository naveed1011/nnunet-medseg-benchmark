"""
Build the nnU-Net vs Swin UNETR comparison table, figures and a results page.

Inputs (any subset, one `--arm name=path` each):
  * nnU-Net arm  : a folder with `per_case_metrics.csv` (from `python -m src.metrics`),
                   OR nnU-Net's own `validation/summary.json`.
  * Swin arm     : a folder with `per_case_metrics.csv`.

Outputs (into --out):
  * comparison_table.csv / .md        — side-by-side means (+std) per label
  * comparison_summary.json           — machine-readable version of the table
  * figures/boxplot_per_case_dice.png — per-case distributions (variance is the point)
  * figures/bar_dice_by_label.png     — mean ± std per label
  * figures/scatter_dice_vs_hd95.png  — accuracy vs boundary error (when HD95 exists)
  * RESULTS.md                        — ready-to-commit results page with the table
                                        and the figures referenced relative to it

CLI
---
    python -m src.compare \
        --arm nnunet=results/brats_nnunet \
        --arm swinunetr=results/brats_swinunetr \
        --out docs/results/brats --task "BraTS 2021 subset (NCR/ED/ET)"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# stable colours so every figure in the repo uses the same model colour
PALETTE = ["#2E86AB", "#E4572E", "#17BEBB", "#8E6C88", "#F4A259", "#5B8E7D"]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _empty_arm() -> pd.DataFrame:
    return pd.DataFrame(columns=["case", "label", "dice", "hd95_mm"])


def _from_nnunet_summary(path: Path) -> pd.DataFrame:
    """nnU-Net's validation/summary.json -> tidy rows (Dice only)."""
    data = json.loads(path.read_text())
    rows = []
    for entry in data.get("metric_per_case", []):
        ref = entry.get("reference_file") or entry.get("reference") or "?"
        case = Path(str(ref)).name
        for suffix in (".nii.gz", ".nii"):
            if case.endswith(suffix):
                case = case[: -len(suffix)]
        for lb, metrics in entry.get("metrics", {}).items():
            rows.append({
                "case": case,
                "label": int(lb),
                "dice": float(metrics.get("Dice", np.nan)),
                "hd95_mm": float(metrics.get("HD95", np.nan)),
            })
    return pd.DataFrame(rows)


def load_arm(path: str | Path) -> pd.DataFrame:
    """Load per-case metrics from a folder/file in any format we produce."""
    path = Path(path)
    if path.is_dir():
        if (path / "per_case_metrics.csv").exists():
            path = path / "per_case_metrics.csv"
        elif (path / "validation" / "summary.json").exists():
            path = path / "validation" / "summary.json"
        elif (path / "summary.json").exists():
            path = path / "summary.json"
        else:
            raise SystemExit(
                f"No per_case_metrics.csv or summary.json in {path}. "
                "Run `python -m src.metrics --pred ... --gt ... --out ...` first."
            )

    if path.suffix == ".csv":
        df = pd.read_csv(path)
        for col in ("case", "label", "dice", "hd95_mm"):
            if col not in df.columns:
                df[col] = np.nan
        df["case"] = df["case"].astype(str)
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
        df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
        df["hd95_mm"] = pd.to_numeric(df["hd95_mm"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        return df[["case", "label", "dice", "hd95_mm"]]

    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if "metric_per_case" in data:                      # nnU-Net's own summary
            return _from_nnunet_summary(path)
        if "per_case" in data:                             # EvalReport written as json
            return pd.DataFrame(data["per_case"])[["case", "label", "dice", "hd95_mm"]]

    raise SystemExit(f"Could not parse metrics at {path}")


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #

def build_table(arms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for arm, df in arms.items():
        for lb, g in df.groupby("label"):
            dice = g["dice"].astype(float)
            hd = g["hd95_mm"].astype(float)
            hd_finite = hd.replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "model": arm,
                "label": int(lb),
                "n_cases": int(g["case"].nunique()),
                "dice_mean": round(float(np.nanmean(dice)), 4),
                "dice_std": round(float(np.nanstd(dice)), 4),
                "dice_median": round(float(np.nanmedian(dice)), 4),
                "dice_min": round(float(np.nanmin(dice)), 4),
                "hd95_mm_mean": round(float(hd_finite.mean()), 3) if len(hd_finite) else np.nan,
                "hd95_mm_p90": round(float(np.percentile(hd_finite, 90)), 3) if len(hd_finite) else np.nan,
                "pct_cases_hd95_inf": round(float(100 * hd.isna().mean()), 1),
            })
    table = pd.DataFrame(rows).sort_values(["label", "model"]).reset_index(drop=True)
    return table


def _table_markdown(table: pd.DataFrame) -> str:
    """Small markdown writer so we don't depend on `tabulate`."""
    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.4g}"
        return str(v)

    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _colors(arms: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {arm: PALETTE[i % len(PALETTE)] for i, arm in enumerate(arms)}


def make_figures(arms: dict[str, pd.DataFrame], out: Path, task: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = _colors(arms)
    labels = sorted({int(lb) for df in arms.values() for lb in df["label"].unique()})
    written: list[Path] = []

    # 1) per-case Dice distribution -----------------------------------------
    fig, axes = plt.subplots(1, len(labels), figsize=(4.6 * len(labels), 4.4), squeeze=False)
    for i, lb in enumerate(labels):
        ax = axes[0][i]
        data, positions, ticks = [], [], []
        for j, (arm, df) in enumerate(arms.items()):
            vals = df[df["label"] == lb]["dice"].astype(float).dropna().values
            if len(vals):
                data.append(vals)
                positions.append(j)
                ticks.append(arm)
        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                            showfliers=False, medianprops={"color": "black", "linewidth": 1.2})
            for patch, arm in zip(bp["boxes"], ticks):
                patch.set_facecolor(colors[arm])
                patch.set_alpha(0.75)
            rng = np.random.default_rng(0)
            for vals, pos in zip(data, positions):
                ax.scatter(pos + rng.uniform(-0.12, 0.12, len(vals)), vals,
                           s=16, color="black", alpha=0.55, zorder=3, linewidths=0)
        ax.set_title(f"Label {lb}")
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=12)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Per-case Dice — {task}  (dots = individual cases)", fontsize=12)
    fig.tight_layout()
    p = fig_dir / "boxplot_per_case_dice.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 2) mean ± std bars -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(labels)), 4.2))
    n_arms = len(arms)
    width = 0.8 / max(n_arms, 1)
    for k, (arm, df) in enumerate(arms.items()):
        means, stds, xs = [], [], []
        for i, lb in enumerate(labels):
            vals = df[df["label"] == lb]["dice"].astype(float).dropna().values
            means.append(float(np.mean(vals)) if len(vals) else 0.0)
            stds.append(float(np.std(vals)) if len(vals) else 0.0)
            xs.append(i + (k - (n_arms - 1) / 2) * width)
        ax.bar(xs, means, width=width, yerr=stds, capsize=4, label=arm,
               color=colors[arm], alpha=0.85, error_kw={"elinewidth": 1, "ecolor": "black"})
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"Label {lb}" for lb in labels])
    ax.set_ylabel("Dice (mean ± std)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False)
    ax.set_title(task)
    fig.tight_layout()
    p = fig_dir / "bar_dice_by_label.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(p)

    # 3) Dice vs HD95 scatter (only if HD95 exists) -------------------------
    has_hd = any(np.isfinite(df["hd95_mm"].astype(float)).any() for df in arms.values())
    if has_hd:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        for arm, df in arms.items():
            for lb in labels:
                sub = df[df["label"] == lb]
                x = sub["hd95_mm"].astype(float).replace([np.inf], np.nan)
                y = sub["dice"].astype(float)
                mask = np.isfinite(x) & np.isfinite(y)
                ax.scatter(x[mask], y[mask], s=22, alpha=0.65, color=colors[arm],
                           label=arm if lb == labels[0] else None)
        ax.set_xlabel("HD95 (mm)  — lower is better")
        ax.set_ylabel("Dice  — higher is better")
        ax.set_title(f"Dice vs boundary error — {task}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        p = fig_dir / "scatter_dice_vs_hd95.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    for p in written:
        print(f"🖼️  Saved {p}")
    return written


# --------------------------------------------------------------------------- #
# results page
# --------------------------------------------------------------------------- #

def _auto_observations(table: pd.DataFrame) -> list[str]:
    """Facts only (no interpretation) so the writeup stays honest."""
    lines: list[str] = []
    for lb, g in table.groupby("label"):
        g = g.sort_values("dice_mean", ascending=False)
        if len(g) < 2:
            continue
        best, runner = g.iloc[0], g.iloc[1]
        delta = best["dice_mean"] - runner["dice_mean"]
        lines.append(
            f"- **Label {lb}:** best mean Dice is `{best['model']}` "
            f"({best['dice_mean']:.3f} ± {best['dice_std']:.3f}), "
            f"{'+' if delta >= 0 else ''}{delta:.3f} vs `{runner['model']}` "
            f"({runner['dice_mean']:.3f})."
        )
    if lines:
        overall_best = table.sort_values("dice_mean", ascending=False).iloc[0]
        lines.append(
            f"- **Overall:** `{overall_best['model']}` has the highest single-label mean "
            f"({overall_best['dice_mean']:.3f} on label {overall_best['label']})."
        )
        lines.append(
            "- Means hide tails — check the boxplot for cases where a model collapses "
            "completely, and the HD95 column for boundary failures."
        )
    return lines


def write_results_page(table: pd.DataFrame, arms: dict[str, pd.DataFrame], out: Path,
                       task: str, figures: list[Path]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_cases = {arm: int(df["case"].nunique()) for arm, df in arms.items()}
    setup = ", ".join(f"`{a}`: {n} cases" for a, n in n_cases.items())

    parts = [
        f"# Results — {task}",
        "",
        f"_Generated by `python -m src.compare` on {stamp}. Do not hand-edit: rerun the command._",
        "",
        f"Evaluation set: {setup}. Metrics: Dice (higher is better), HD95 in mm "
        "(lower is better; `inf` = structure missed entirely).",
        "",
        "## Table",
        "",
        _table_markdown(table),
        "",
        "## Figures",
        "",
    ]
    for p in figures:
        parts += [f"### {p.stem.replace('_', ' ')}", "", f"![{p.stem}](figures/{p.name})", ""]

    obs = _auto_observations(table)
    if obs:
        parts += ["## What the numbers say", ""] + obs + [""]

    parts += [
        "## Read this before quoting these numbers",
        "",
        "- Reduced epoch budget / subset size: these are *pipeline-validated* numbers, "
        "not challenge-leaderboard numbers.",
        "- Single fold, no 5-fold ensemble, no post-processing unless stated.",
        "- nnU-Net self-configures patch/batch/augmentation; the Transformer arm uses fixed "
        "defaults. The comparison is budget-matched, not tuning-matched.",
        "",
    ]
    path = out / "RESULTS.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    print(f"📝 Saved {path}")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True,
                   help="name=path/to/results — repeat per model (e.g. --arm nnunet=results/x)")
    p.add_argument("--out", default="docs/results/comparison")
    p.add_argument("--task", default="3D medical segmentation")
    p.add_argument("--no-figures", action="store_true", help="Table + RESULTS.md only")
    args = p.parse_args()

    arms: dict[str, pd.DataFrame] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"Bad --arm spec '{spec}', expected name=path")
        name = name.strip()
        arms[name] = load_arm(path.strip())
        if arms[name].empty:
            raise SystemExit(f"Arm '{name}' loaded 0 rows from {path}")
        print(f"✅ Loaded arm '{name}' — {len(arms[name])} rows, "
              f"{arms[name]['case'].nunique()} cases from {path}")

    table = build_table(arms)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison_table.csv", index=False)
    (out / "comparison_table.md").write_text(_table_markdown(table) + "\n", encoding="utf-8")
    table.to_json(out / "comparison_summary.json", orient="records", indent=2)

    figures = [] if args.no_figures else make_figures(arms, out, args.task)
    write_results_page(table, arms, out, args.task, figures)

    print("\n" + "=" * 78)
    print(table.to_string(index=False))
    print("=" * 78)
    print(f"💾 Artifacts saved to {out}/")


if __name__ == "__main__":
    main()
