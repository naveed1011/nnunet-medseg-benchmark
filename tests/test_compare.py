"""Tests for the comparison table / figures / results page."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.compare import _auto_observations, build_table, load_arm, main as compare_main


def test_load_arm_reads_our_csv(two_arm_csvs):
    df = load_arm(two_arm_csvs["nnunet"])
    assert list(df.columns) == ["case", "label", "dice", "hd95_mm"]
    assert len(df) == 18
    assert set(df["label"]) == {1, 2, 3}


def test_load_arm_reads_directory_and_file(two_arm_csvs):
    as_dir = load_arm(two_arm_csvs["swinunetr"])
    as_file = load_arm(two_arm_csvs["swinunetr"] / "per_case_metrics.csv")
    pd.testing.assert_frame_equal(as_dir, as_file)


def test_load_arm_reads_nnunet_summary(tmp_path):
    summary = {
        "metric_per_case": [
            {"reference_file": "/x/case_0.nii.gz",
             "prediction_file": "/x/case_0.nii.gz",
             "metrics": {"1": {"Dice": 0.91}, "2": {"Dice": 0.72}}},
            {"reference_file": "/x/case_1.nii.gz",
             "prediction_file": "/x/case_1.nii.gz",
             "metrics": {"1": {"Dice": 0.83}, "2": {"Dice": 0.61}}},
        ]
    }
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary))
    df = load_arm(p)
    assert len(df) == 4
    assert df[df["label"] == 1]["dice"].mean() == pytest.approx(0.87)
    assert df["hd95_mm"].isna().all()


def test_load_arm_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="No per_case_metrics.csv"):
        load_arm(tmp_path)


def test_build_table_shape_and_values(two_arm_csvs):
    arms = {name: load_arm(path) for name, path in two_arm_csvs.items()}
    table = build_table(arms)
    assert set(table.columns) >= {"model", "label", "n_cases", "dice_mean", "dice_std",
                                  "hd95_mm_mean", "pct_cases_hd95_inf"}
    assert len(table) == 6                     # 2 arms x 3 labels
    assert table["n_cases"].eq(6).all()
    # nnU-Net arm was generated 0.10 Dice better on average
    nn = table[table["model"] == "nnunet"]["dice_mean"].mean()
    sw = table[table["model"] == "swinunetr"]["dice_mean"].mean()
    assert nn > sw


def test_auto_observations_are_factual(two_arm_csvs):
    table = build_table({n: load_arm(p) for n, p in two_arm_csvs.items()})
    lines = _auto_observations(table)
    assert any("best mean Dice" in line for line in lines)
    assert any(line.startswith("- **Label 1:**") for line in lines)


def test_figures_and_results_page(two_arm_csvs, tmp_path):
    arms = {name: load_arm(path) for name, path in two_arm_csvs.items()}
    out = tmp_path / "comparison"
    out.mkdir()
    table = build_table(arms)

    from src.compare import make_figures, write_results_page

    figures = make_figures(arms, out, "unit-test task")
    names = {p.name for p in figures}
    assert "boxplot_per_case_dice.png" in names
    assert "bar_dice_by_label.png" in names
    assert all(p.exists() and p.stat().st_size > 1000 for p in figures)

    page = write_results_page(table, arms, out, "unit-test task", figures)
    text = page.read_text(encoding="utf-8")
    assert "# Results — unit-test task" in text
    assert "figures/boxplot_per_case_dice.png" in text
    assert "| model | label |" in text
    assert "not challenge-leaderboard numbers" in text


def test_cli_end_to_end(two_arm_csvs, tmp_path, monkeypatch):
    out = tmp_path / "cli-out"
    argv = [
        "compare",
        "--arm", f"nnunet={two_arm_csvs['nnunet']}",
        "--arm", f"swinunetr={two_arm_csvs['swinunetr']}",
        "--out", str(out),
        "--task", "cli task",
    ]
    monkeypatch.setattr("sys.argv", argv)
    compare_main()
    assert (out / "comparison_table.csv").exists()
    assert (out / "RESULTS.md").exists()
    assert (out / "figures" / "boxplot_per_case_dice.png").exists()
