# Outputs

This folder is used for generated inference artifacts:

- `sample_predictions.csv`
- annotated images under `sample_detection_images/`
- `training_summary.md`
- `requirements_checklist.md`

Generated images and prediction files can be reproduced with:

```powershell
.\.venv-gpu\Scripts\mps-infer.exe `
  --input data/bmd45_yolo_gpu/images/val `
  --weights models/bmd45_yolo11s_gpu_50epochs_best.pt `
  --output-dir outputs `
  --conf 0.25 `
  --imgsz 640 `
  --max-sample-images 10
```
