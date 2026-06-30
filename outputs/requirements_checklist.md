# Project Requirements Checklist

This file maps the project implementation to the assessment requirements from the traffic-density problem statement.

| Requirement | Where it is handled | Status |
| --- | --- | --- |
| Build a small app for traffic-camera images | `app/streamlit_app.py` | Done |
| Allow image upload | Streamlit uploader accepts one or more `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` files | Done |
| Run vehicle detection | `src/mps_traffic_density/detector.py` loads the adapted YOLO model and runs inference | Done |
| Draw detection boxes and class labels | `src/mps_traffic_density/visualization.py` and the Streamlit preview | Done |
| Count detected vehicles by class | `detector.py` returns class-wise counts; app displays them as a table | Done |
| Assign a density label | `src/mps_traffic_density/density.py` maps total vehicle count to `unclear`, `low`, `medium`, or `high` | Done |
| Support folder/batch inference | `mps-infer` CLI supports image files or folders | Done |
| Provide CSV output | `outputs/sample_predictions.csv` | Done |
| Provide annotated sample images | `outputs/sample_detection_images/` | Done |
| Use BMD-45 traffic dataset | `data/bmd45_yolo_gpu` prepared from BMD-45 | Done |
| Run short adaptation training | YOLO11s adapted on 1000 train / 200 validation images for 50 epochs | Done |
| Include training details and metrics | `outputs/training_summary.md` and `README.md` | Done |
| Compare or discuss transformer option | RT-DETR-L comparison documented in `outputs/training_summary.md` and `README.md` | Done |
| Keep repository reproducible | `pyproject.toml`, `scripts/run_adapted.ps1`, `mps-prepare-bmd45`, `mps-train`, and `mps-infer` | Done |
| Mention limitations honestly | `README.md` Known Limitations section | Done |
| Keep documentation concise and reproducible | `README.md` and `outputs/training_summary.md` | Done |

## Final Model Used By The App

The Streamlit app uses the adapted YOLO11s checkpoint by default:

```text
models/bmd45_yolo11s_gpu_50epochs_best.pt
```

If that packaged model is missing, the app falls back to the local Ultralytics run checkpoint:

```text
runs/detect/runs/detect/bmd45_yolo11s_gpu_50epochs/weights/best.pt
```

## Final Reported Validation Result

YOLO11s on the BMD-45 GPU subset:

- train images: 1000
- validation images: 200
- epochs: 50
- image size: 640
- batch size: 16
- device: RTX 4060 Laptop GPU
- mAP50: 0.583
- mAP50-95: 0.473
- precision: 0.567
- recall: 0.584
