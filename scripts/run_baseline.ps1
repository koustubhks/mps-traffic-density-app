param(
    [string]$InputPath = "data/demo_images",
    [string]$Weights = "yolo11n.pt"
)

python -m uv run mps-infer --input $InputPath --weights $Weights --output-dir outputs
