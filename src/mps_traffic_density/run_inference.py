from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .detector import load_yolo_model, predict_image
from .density import LOW_MAX_COUNT, MEDIUM_MAX_COUNT
from .io_utils import collect_images, save_predictions_json, save_summary_csv
from .visualization import draw_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vehicle detection and density labeling.")
    parser.add_argument("--input", required=True, help="Path to one image or a folder of images.")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO weights, e.g. yolo11n.pt or runs/.../best.pt.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for predictions and annotated images.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--max-sample-images", type=int, default=10, help="Maximum annotated images to save.")
    parser.add_argument("--low-max-count", type=int, default=LOW_MAX_COUNT, help="Max count for low density.")
    parser.add_argument("--medium-max-count", type=int, default=MEDIUM_MAX_COUNT, help="Max count for medium density.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "sample_detection_images"
    images = collect_images(args.input)
    if not images:
        raise SystemExit(f"No images found under {args.input}")

    model = load_yolo_model(args.weights)
    predictions = []

    for image_path in tqdm(images, desc="Running inference"):
        prediction = predict_image(
            image_path,
            model=model,
            weights=args.weights,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            low_max_count=args.low_max_count,
            medium_max_count=args.medium_max_count,
        )
        predictions.append(prediction)

    sample_predictions = sorted(
        predictions,
        key=lambda prediction: (prediction.total_vehicles, prediction.mean_vehicle_confidence or 0.0),
        reverse=True,
    )[: args.max_sample_images]
    annotated_paths: dict[Path, str] = {}
    sample_dir.mkdir(parents=True, exist_ok=True)
    for old_sample in sample_dir.glob("*_detections.jpg"):
        old_sample.unlink()

    for prediction in sample_predictions:
        annotated = sample_dir / f"{prediction.image_path.stem}_detections.jpg"
        draw_prediction(prediction, annotated)
        annotated_paths[prediction.image_path] = str(annotated)

    rows = [
        prediction.to_summary_row(annotated_paths.get(prediction.image_path, ""))
        for prediction in predictions
    ]

    csv_path = save_summary_csv(rows, output_dir / "sample_predictions.csv")
    json_path = save_predictions_json(predictions, output_dir / "sample_predictions.json")
    print(f"Processed {len(predictions)} image(s).")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Annotated samples: {sample_dir}")


if __name__ == "__main__":
    main()
