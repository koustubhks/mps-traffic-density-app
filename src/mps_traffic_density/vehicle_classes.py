from __future__ import annotations

VEHICLE_CLASS_KEYWORDS = (
    "auto",
    "bicycle",
    "bike",
    "bus",
    "car",
    "cycle",
    "hmv",
    "lcV".lower(),
    "lorry",
    "motor",
    "scooter",
    "three",
    "tractor",
    "truck",
    "van",
    "vehicle",
)

NON_VEHICLE_CLASSES = {
    "person",
    "traffic light",
    "stop sign",
    "parking meter",
    "bench",
}


def is_vehicle_class(class_name: str) -> bool:
    normalized = class_name.strip().lower().replace("_", " ").replace("-", " ")
    if normalized in NON_VEHICLE_CLASSES:
        return False
    return any(keyword in normalized for keyword in VEHICLE_CLASS_KEYWORDS)


def normalize_class_name(class_name: str) -> str:
    return class_name.strip().replace("_", " ").replace("-", " ").lower()

