from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .density import density_from_vehicle_count
from .schema import Detection, ImagePrediction
from .vehicle_classes import is_vehicle_class, normalize_class_name


def load_yolo_model(weights: str | Path = "yolo11n.pt") -> Any:
    """Load a YOLO model lazily so non-model utilities stay lightweight."""
    from ultralytics import YOLO

    return YOLO(str(weights))


def predict_image(
    image_path: str | Path,
    *,
    weights: str | Path = "yolo11n.pt",
    model: Any | None = None,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    low_max_count: int = 7,
    medium_max_count: int = 18,
) -> ImagePrediction:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        width, height = image.size

    yolo_model = model or load_yolo_model(weights)
    predict_kwargs: dict[str, Any] = {
        "source": str(image_path),
        "conf": conf,
        "imgsz": imgsz,
        "save": False,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device

    result = yolo_model.predict(**predict_kwargs)[0]
    names = result.names or {}
    detections: list[Detection] = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for box, class_id_raw, confidence in zip(boxes, classes, confidences):
            class_id = int(class_id_raw)
            class_name = names.get(class_id, str(class_id))
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(class_name),
                    confidence=float(confidence),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    is_vehicle=is_vehicle_class(str(class_name)),
                )
            )

    vehicle_counts: dict[str, int] = {}
    for detection in detections:
        if not detection.is_vehicle:
            continue
        name = normalize_class_name(detection.class_name)
        vehicle_counts[name] = vehicle_counts.get(name, 0) + 1

    total_vehicles = sum(vehicle_counts.values())
    density_label, density_note = density_from_vehicle_count(
        total_vehicles,
        low_max_count=low_max_count,
        medium_max_count=medium_max_count,
    )

    return ImagePrediction(
        image_path=image_path,
        width=width,
        height=height,
        detections=detections,
        vehicle_counts=dict(sorted(vehicle_counts.items())),
        total_vehicles=total_vehicles,
        density_label=density_label,
        density_note=density_note,
    )

