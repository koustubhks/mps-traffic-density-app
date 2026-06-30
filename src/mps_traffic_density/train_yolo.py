from __future__ import annotations

import argparse

from .detector import load_yolo_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short YOLO adaptation on the BMD-45 subset.")
    parser.add_argument("--data", default="data/bmd45_yolo/data.yaml", help="YOLO data.yaml path.")
    parser.add_argument("--model", default="yolo11n.pt", help="Starting YOLO model.")
    parser.add_argument("--epochs", type=int, default=5, help="Small adaptation epoch count.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--project", default="runs/detect", help="Ultralytics project output directory.")
    parser.add_argument("--name", default="bmd45_yolo11n_small", help="Ultralytics run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_yolo_model(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

