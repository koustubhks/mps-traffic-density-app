# Traffic-Camera Vehicle Detection and Density Labeling

This project is a Streamlit-based computer vision application for traffic-camera images. It detects vehicles, labels them by class, counts the detected vehicles, and assigns a simple traffic-density label: `unclear`, `low`, `medium`, or `high`.

The main goal is to keep the pipeline practical and reproducible: prepare a small BMD-45 subset, adapt a detector, run inference, generate outputs, and provide an app that can be tested with new traffic images.

## Features

- Upload one or more traffic images through a Streamlit app
- Detect and label vehicles with bounding boxes
- Count detected vehicles by class
- Assign a count-based density label
- Export batch results as CSV
- Run image or folder inference from the command line
- Compare a YOLO detector with a transformer-based detector

## Dataset

Dataset: [BMD-45: Bengaluru Mobility Dataset](https://huggingface.co/datasets/iisc-aim/BMD-45)

BMD-45 contains CCTV traffic images with vehicle bounding-box annotations. The full dataset is not committed to this repository. It can be reproduced locally using the preparation command below.

Final YOLO subset:

- 1000 training images
- 200 validation/demo images

RT-DETR comparison subset:

- 300 training images
- 80 validation/demo images

## Models Used

Three model runs were used during development:

| Model | Purpose | Train Images | Val Images | Epochs | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | CPU baseline | 120 | 30 | 20 | 0.204 | 0.148 | 0.521 | 0.185 |
| YOLO11s | Final app model | 1000 | 200 | 50 | 0.583 | 0.473 | 0.567 | 0.584 |
| RT-DETR-L | Transformer comparison | 300 | 80 | 30 | 0.491 | 0.407 | 0.474 | 0.516 |

YOLO11s was selected for the app because it gave the best validation performance while still being lightweight enough for an interactive demo.

RT-DETR-L was included as the transformer-based comparison model. It performed competitively, especially considering it was trained on a smaller subset due to local GPU memory constraints.

When Ultralytics is installed locally, the Streamlit app also shows COCO-pretrained YOLO options. These are useful for general web or phone traffic photos, where the BMD-adapted model may miss small distant vehicles. The public Streamlit Cloud app uses the exported ONNX model for a smaller and more reliable deployment runtime.

## Local Setup

Install `uv` if needed:

```powershell
python -m pip install uv
```

Create the environment:

```powershell
cd E:\MPS_Traffic_Density_App
python -m uv sync
```

For GPU training on Windows, I used a separate CUDA-enabled environment:

```powershell
python -m uv venv .venv-gpu --python 3.12
.\.venv-gpu\Scripts\python.exe -m ensurepip --upgrade
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gpu\Scripts\python.exe -m pip install -e .
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

GPU check:

```powershell
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Run the App

```powershell
python -m uv run streamlit run app/streamlit_app.py
```

For deployment platforms that expect a root app file:

```powershell
streamlit run app.py
```

The app loads this adapted model by default:

```text
models/bmd45_yolo11s_gpu_50epochs_best.onnx
```

The ONNX model is used for Streamlit Cloud deployment because it avoids the heavier PyTorch/Ultralytics runtime and makes the public demo more reliable. The PyTorch `.pt` checkpoint is still included for local experiments and training continuation.

## Prepare the Dataset

Final YOLO subset:

```powershell
.\.venv-gpu\Scripts\mps-prepare-bmd45.exe `
  --train-count 1000 `
  --val-count 200 `
  --output data/bmd45_yolo_gpu `
  --cache-dir data/bmd45_raw_gpu
```

RT-DETR subset:

```powershell
.\.venv-gpu\Scripts\mps-prepare-bmd45.exe `
  --train-count 300 `
  --val-count 80 `
  --output data/bmd45_yolo_rtdetr `
  --cache-dir data/bmd45_raw_gpu
```

## Train the Final YOLO11s Model

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

`workers=0` was used to avoid PyTorch/OpenCV multiprocessing issues on Windows.

The final model was copied to:

```text
models/bmd45_yolo11s_gpu_50epochs_best.pt
```

## Train the RT-DETR Transformer Comparison

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

RT-DETR-L is included to show how a transformer-style detector compares with the YOLO baseline on the same traffic-density task.

## Run Inference

```powershell
.\.venv-gpu\Scripts\mps-infer.exe `
  --input data/bmd45_yolo_gpu/images/val `
  --weights models/bmd45_yolo11s_gpu_50epochs_best.pt `
  --output-dir outputs `
  --conf 0.25 `
  --imgsz 640 `
  --max-sample-images 10
```

Generated outputs:

- `outputs/sample_predictions.csv`
- `outputs/sample_detection_images/`
- `outputs/training_summary.md`
- `outputs/requirements_checklist.md`

## Density Logic

The density label is based on the total number of detected vehicles:

- `unclear`: 0 detected vehicles
- `low`: 1-5 detected vehicles
- `medium`: 6-12 detected vehicles
- `high`: more than 12 detected vehicles

This is an explainable first-pass rule rather than a calibrated traffic-engineering metric. In a production system, density should also consider the visible road area, lane count, camera angle, and road-region mask.

## Repository Structure

```text
README.md
requirements.txt
pyproject.toml
app.py
app/
  streamlit_app.py
data/
  README.md
  demo_images/
models/
  bmd45_yolo11s_gpu_50epochs_best.onnx
  bmd45_yolo11s_gpu_50epochs_best.pt
outputs/
  sample_predictions.csv
  sample_detection_images/
  requirements_checklist.md
  training_summary.md
docs/
  llm_usage.md
scripts/
  run_adapted.ps1
  run_baseline.ps1
src/
  mps_traffic_density/
    detector.py
    density.py
    io_utils.py
    onnx_detector.py
    prepare_bmd45_subset.py
    run_inference.py
    schema.py
    train_yolo.py
    vehicle_classes.py
    visualization.py
```

## Deployment

This repository is prepared for Streamlit Community Cloud.

Recommended settings:

- Repository: `koustubhks/mps-traffic-density-app`
- Branch: `main`
- App file: `app.py`
- Python version: `3.12` from Advanced settings
- Python requirements: `requirements.txt`
- Keep `models/bmd45_yolo11s_gpu_50epochs_best.onnx` in the repository
- Do not upload local dataset folders, training runs, virtual environments, or raw downloaded checkpoints

## Limitations

- Density is based on detected vehicle count only.
- The app does not yet use road segmentation, lane geometry, or visible-road-area normalization.
- Small, far-away, occluded, or heavily compressed vehicles may be missed.
- The BMD-adapted model is strongest on CCTV-style traffic images and may underperform on stock photos or phone images.
- RT-DETR-L was trained on a smaller subset than YOLO11s because of local memory limits.
- Night scenes, glare, rain, blur, and dense occlusion can reduce detection quality.

## Future Work

- Add road-region segmentation so detections outside the roadway are ignored.
- Estimate density using vehicle box area or road occupancy, not count alone.
- Normalize density by lane count and visible road area.
- Train on a larger and more diverse traffic dataset.
- Compare stronger transformer detectors such as DINO or Deformable DETR.
- Explore parameter-efficient adaptation for transformer detectors.
- Add video input support for traffic clips and frame-level density trends.

## Development Assistance

See `docs/llm_usage.md` for the development-assistance note requested in the problem statement.
