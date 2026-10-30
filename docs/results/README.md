# Results

Everything in this folder is not currently the results as they are pending and I will do it soon.

```
RESULTS.md                     # table + auto-generated observations + caveats
comparison_table.csv / .md     # per-label means
comparison_summary.json        # machine-readable
figures/boxplot_per_case_dice.png
figures/bar_dice_by_label.png
figures/scatter_dice_vs_hd95.png
```

I will Regenerate them from the per-case CSVs:

```bash
python -m src.compare \
  --arm nnunet=path/to/brats_nnunet \
  --arm swinunetr=path/to/brats_swinunetr \
  --out docs/results/brats --task "BraTS 2021 subset (NCR/ED/ET)"
```

| Folder | What it is |
|---|---|
| `demo/` | Synthetic-phantom smoke test (`python run_pipeline.py demo`). Proves the pipeline works and the results are; **not** a medical result yet. |
| `brats/` | The real comparison, produced by `notebooks/04_compare_nnunet_vs_swin.ipynb`. |

Numbers in `RESULTS.md` are per-case currently which means over held-out volumes only.
