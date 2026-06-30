# Data

This repository keeps only a small demo image subset for reproducible inference.

The full BMD-45 dataset should not be committed. Use the preparation script to create a local subset:

```powershell
.\.venv-gpu\Scripts\mps-prepare-bmd45.exe `
  --train-count 1000 `
  --val-count 200 `
  --output data/bmd45_yolo_gpu `
  --cache-dir data/bmd45_raw_gpu
```
