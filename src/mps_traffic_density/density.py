from __future__ import annotations


LOW_MAX_COUNT = 5
MEDIUM_MAX_COUNT = 12


def density_from_vehicle_count(
    total_vehicles: int,
    *,
    low_max_count: int = LOW_MAX_COUNT,
    medium_max_count: int = MEDIUM_MAX_COUNT,
) -> tuple[str, str]:
    """Convert a detected vehicle count into a transparent density label.

    The problem statement asks for a simple, explainable label rather than a
    calibrated traffic engineering metric. These thresholds are intentionally
    count-based so the app remains interpretable during review.
    """
    if total_vehicles <= 0:
        return (
            "unclear",
            "No vehicles were detected. The scene may be empty, dark, occluded, or outside the model domain.",
        )
    if total_vehicles <= low_max_count:
        return "low", f"Detected {total_vehicles} vehicles, at or below the low threshold of {low_max_count}."
    if total_vehicles <= medium_max_count:
        return (
            "medium",
            f"Detected {total_vehicles} vehicles, between {low_max_count + 1} and {medium_max_count}.",
        )
    return "high", f"Detected {total_vehicles} vehicles, above the high threshold of {medium_max_count}."


def density_threshold_description(
    *,
    low_max_count: int = LOW_MAX_COUNT,
    medium_max_count: int = MEDIUM_MAX_COUNT,
) -> str:
    return (
        f"unclear: 0 detected vehicles; low: 1-{low_max_count}; "
        f"medium: {low_max_count + 1}-{medium_max_count}; high: >{medium_max_count}."
    )
