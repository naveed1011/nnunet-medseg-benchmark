# Results

Everything in this folder is **generated**, not hand-written. Each run folder contains:

```
RESULTS.md                     # table + auto-generated observations + caveats
comparison_table.csv / .md     # per-label means
comparison_summary.json        # machine-readable
figures/boxplot_per_case_dice.png
figures/bar_dice_by_label.png
figures/scatter_dice_vs_hd95.png
```

Regenerate any of them from the per-case CSVs:

```bash
python -m src.compare \
  --arm nnunet=path/to/brats_nnunet \
  --arm swinunetr=path/to/brats_swinunetr \
  --out docs/results/brats --task "BraTS 2021 subset (NCR/ED/ET)"
```

| Folder | What it is |
|---|---|
| `demo/` | Synthetic-phantom smoke test (`python run_pipeline.py demo`). Proves the pipeline; **not** a medical result. |
| `brats/` | The real comparison, produced by `notebooks/04_compare_nnunet_vs_swin.ipynb`. |

Numbers in `RESULTS.md` are per-case means over held-out volumes only. Read the caveats at the
bottom of that file before quoting anything.
