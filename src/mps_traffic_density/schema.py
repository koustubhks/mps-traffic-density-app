from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    is_vehicle: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImagePrediction:
    image_path: Path
    width: int
    height: int
    detections: list[Detection]
    vehicle_counts: dict[str, int]
    total_vehicles: int
    density_label: str
    density_note: str

    @property
    def mean_vehicle_confidence(self) -> float | None:
        confidences = [d.confidence for d in self.detections if d.is_vehicle]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "detections": [d.to_dict() for d in self.detections],
            "vehicle_counts": self.vehicle_counts,
            "total_vehicles": self.total_vehicles,
            "density_label": self.density_label,
            "density_note": self.density_note,
            "mean_vehicle_confidence": self.mean_vehicle_confidence,
        }

    def to_summary_row(self, annotated_image: str | None = None) -> dict[str, Any]:
        import json

        return {
            "image_path": str(self.image_path),
            "total_vehicles": self.total_vehicles,
            "density_label": self.density_label,
            "mean_confidence": self.mean_vehicle_confidence,
            "counts_json": json.dumps(self.vehicle_counts, sort_keys=True),
            "annotated_image": annotated_image or "",
        }

