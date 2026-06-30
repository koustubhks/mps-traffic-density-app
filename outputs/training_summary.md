# Training Summary

This submission uses a GPU-trained YOLO11s model as the final detector and includes RT-DETR-L as a transformer-based comparison.

## Dataset Subsets

Source dataset: `iisc-aim/BMD-45` from Hugging Face.

Final YOLO11s subset:

- Training images: 1000
- Validation/demo images: 200
- Local dataset path: `data/bmd45_yolo_gpu`

RT-DETR-L comparison subset:

- Training images: 300
- Validation/demo images: 80
- Local dataset path: `data/bmd45_yolo_rtdetr`

## Final YOLO11s Adaptation

```powershell
.\.venv-gpu\Scripts\yolo.exe detect train `
  model=yolo11s.pt `
  data=data/bmd45_yolo_gpu/data.yaml `
  epochs=50 `
  imgsz=640 `
  batch=16 `
  device=0 `
  workers=0 `
  cache=False `
  project=runs/detect `
  name=bmd45_yolo11s_gpu_50epochs
```

Final weights:

```text
models/bmd45_yolo11s_gpu_50epochs_best.pt
```

Validation result:

- mAP50: 0.583
- mAP50-95: 0.473
- Precision: 0.567
- Recall: 0.584
- Device: NVIDIA GeForce RTX 4060 Laptop GPU

## RT-DETR-L Transformer Comparison

```powershell
.\.venv-gpu\Scripts\yolo.exe detect train `
  model=rtdetr-l.pt `
  data=data/bmd45_yolo_rtdetr/data.yaml `
  epochs=30 `
  imgsz=512 `
  batch=2 `
  device=0 `
  workers=0 `
  cache=False `
  project=runs/detect `
  name=bmd45_rtdetr_l_gpu_20epochs_300
```

Validation result:

- mAP50: 0.491
- mAP50-95: 0.407
- Precision: 0.474
- Recall: 0.516
- Device: NVIDIA GeForce RTX 4060 Laptop GPU

## Comparison

| Model | Train Images | Val Images | Epochs | mAP50 | mAP50-95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11s | 1000 | 200 | 50 | 0.583 | 0.473 | 0.567 | 0.584 |
| RT-DETR-L | 300 | 80 | 30 | 0.491 | 0.407 | 0.474 | 0.516 |
| YOLO11n CPU baseline | 120 | 30 | 20 | 0.204 | 0.148 | 0.521 | 0.185 |

YOLO11s was selected as the final model because it achieved the best overall validation performance. RT-DETR-L was included as a transformer-based comparison and performed competitively, especially considering it was trained on a smaller subset.

## Final Inference Command

```powershell
.\.venv-gpu\Scripts\mps-infer.exe `
  --input data/bmd45_yolo_gpu/images/val `
  --weights models/bmd45_yolo11s_gpu_50epochs_best.pt `
  --output-dir outputs `
  --conf 0.25 `
  --imgsz 640 `
  --max-sample-images 10
```

## Density Thresholds

- `unclear`: 0 detected vehicles
- `low`: 1-5 detected vehicles
- `medium`: 6-12 detected vehicles
- `high`: more than 12 detected vehicles

## Limitations

The density label is count-based and does not yet account for lane geometry, visible road area, road masks, or vehicle occupancy area. Small distant vehicles, glare, blur, and occlusion can reduce detection quality. RT-DETR-L used a smaller dataset subset than YOLO11s due to local memory constraints.

## Future Work

- Add road-region segmentation and ignore detections outside the roadway.
- Normalize density by visible road area and lane count.
- Add a box-area or road-occupancy based density estimate.
- Train and evaluate on a larger, more diverse dataset.
- Compare additional transformer detectors such as DINO or Deformable DETR.
