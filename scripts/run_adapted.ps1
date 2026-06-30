python -m uv run mps-infer `
  --input data/bmd45_yolo_gpu/images/val `
  --weights models/bmd45_yolo11s_gpu_50epochs_best.pt `
  --output-dir outputs `
  --conf 0.25 `
  --imgsz 640 `
  --max-sample-images 10
