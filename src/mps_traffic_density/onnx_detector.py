from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from .density import density_from_vehicle_count
from .schema import Detection, ImagePrediction
from .vehicle_classes import is_vehicle_class, normalize_class_name


BMD45_CLASS_NAMES = {
    0: "Hatchback",
    1: "Sedan",
    2: "SUV",
    3: "MUV",
    4: "Bus",
    5: "Truck",
    6: "Three-wheeler",
    7: "Two-wheeler",
    8: "LCV",
    9: "Mini-bus",
    10: "Tempo-traveller",
    11: "Bicycle",
    12: "Van",
}


class OnnxYoloDetector:
    def __init__(self, weights: str | Path):
        import onnxruntime as ort

        self.weights = Path(weights)
        self.session = ort.InferenceSession(str(self.weights), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = int(input_shape[-1]) if isinstance(input_shape[-1], int) else 640

    def predict(
        self,
        image_path: str | Path,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        low_max_count: int = 7,
        medium_max_count: int = 18,
    ) -> ImagePrediction:
        image_path = Path(image_path)
        image, ratio, pad_x, pad_y = _load_letterboxed_image(image_path, self.input_size)
        outputs = self.session.run(None, {self.input_name: image})[0]
        detections = _decode_yolo_output(outputs, conf, iou, ratio, pad_x, pad_y, image_path)

        with Image.open(image_path) as original:
            width, height = original.size

        detections = [
            replace(
                detection,
                x1=max(0.0, min(float(width), detection.x1)),
                x2=max(0.0, min(float(width), detection.x2)),
                y1=max(0.0, min(float(height), detection.y1)),
                y2=max(0.0, min(float(height), detection.y2)),
            )
            for detection in detections
        ]

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


def _load_letterboxed_image(image_path: Path, input_size: int) -> tuple[np.ndarray, float, float, float]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    ratio = min(input_size / width, input_size / height)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    pad_x = (input_size - new_width) / 2
    pad_y = (input_size - new_height) / 2

    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))[None, :, :, :]
    return array, ratio, pad_x, pad_y


def _decode_yolo_output(
    output: np.ndarray,
    conf: float,
    iou: float,
    ratio: float,
    pad_x: float,
    pad_y: float,
    image_path: Path,
) -> list[Detection]:
    predictions = np.squeeze(output, axis=0)
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep_mask = scores >= conf

    boxes_xywh = boxes_xywh[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    if len(scores) == 0:
        return []

    boxes_xyxy = _xywh_to_original_xyxy(boxes_xywh, ratio, pad_x, pad_y)
    keep_indices = _batched_nms(boxes_xyxy, scores, class_ids, iou)

    detections: list[Detection] = []
    for idx in keep_indices:
        class_id = int(class_ids[idx])
        class_name = BMD45_CLASS_NAMES.get(class_id, str(class_id))
        x1, y1, x2, y2 = boxes_xyxy[idx].tolist()
        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(scores[idx]),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                is_vehicle=is_vehicle_class(class_name),
            )
        )

    return sorted(detections, key=lambda item: item.confidence, reverse=True)


def _xywh_to_original_xyxy(boxes_xywh: np.ndarray, ratio: float, pad_x: float, pad_y: float) -> np.ndarray:
    boxes = boxes_xywh.copy()
    boxes[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_x) / ratio
    boxes[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_y) / ratio
    boxes[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - pad_x) / ratio
    boxes[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pad_y) / ratio
    return boxes


def _batched_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float) -> list[int]:
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        class_keep = _nms(boxes[indices], scores[indices], iou_threshold)
        kept.extend(indices[class_keep].tolist())
    return sorted(kept, key=lambda idx: float(scores[idx]), reverse=True)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])

        inter_width = np.maximum(0.0, xx2 - xx1)
        inter_height = np.maximum(0.0, yy2 - yy1)
        intersection = inter_width * inter_height
        union = areas[current] + areas[remaining] - intersection + 1e-9
        ious = intersection / union
        order = remaining[ious <= iou_threshold]

    return keep
