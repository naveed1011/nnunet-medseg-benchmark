Curated figures referenced from README.md live here. Keep them small (< 1 MB each):
PNG at ~150 dpi, no checkpoints, no datasets.

Expected files after a full run:

  demo_data_qc.png          synthetic phantom: image vs label
  demo_prediction.png       image | prediction | GT | error map (smoke test)
  demo_saliency.png         gradient saliency through nnU-Net (smoke test)
  heart_*.png               MSD Heart curves / predictions
  brats_boxplot.png         per-case Dice, nnU-Net vs Swin UNETR
  xai_*.png                 best/worst-case saliency + GradCAM overlays

These are copies of files produced in demo/ or docs/results/; never commit the
underlying volumes or model weights.
