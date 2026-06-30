from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schema import Detection, ImagePrediction


BOX_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
]


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def draw_prediction(
    prediction: ImagePrediction,
    output_path: str | Path,
    *,
    vehicle_only: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(prediction.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font()

    for detection in prediction.detections:
        if vehicle_only and not detection.is_vehicle:
            continue
        _draw_detection(draw, detection, font)

    banner = (
        f"vehicles={prediction.total_vehicles} | "
        f"density={prediction.density_label}"
    )
    draw.rectangle((0, 0, image.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), banner, fill=(255, 255, 255), font=font)

    image.save(output_path)
    return output_path


def _draw_detection(draw: ImageDraw.ImageDraw, detection: Detection, font: ImageFont.ImageFont) -> None:
    color = BOX_COLORS[detection.class_id % len(BOX_COLORS)]
    x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    label = f"{detection.class_name} {detection.confidence:.2f}"
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_y = max(0, y1 - text_h - 4)
    draw.rectangle((x1, text_y, x1 + text_w + 6, text_y + text_h + 4), fill=color)
    draw.text((x1 + 3, text_y + 2), label, fill=(255, 255, 255), font=font)

