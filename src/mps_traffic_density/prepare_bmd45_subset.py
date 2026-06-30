from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "iisc-aim/BMD-45"
SPLIT_DIRS = {
    "train": "BMD-45-Train",
    "val": "BMD-45-Val",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert a small BMD-45 subset to YOLO format.")
    parser.add_argument("--output", default="data/bmd45_yolo", help="Output YOLO dataset directory.")
    parser.add_argument("--train-count", type=int, default=120, help="Number of training images.")
    parser.add_argument("--val-count", type=int, default=30, help="Number of validation/demo images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection.")
    parser.add_argument("--cache-dir", default="data/bmd45_raw", help="Local Hugging Face cache directory.")
    return parser.parse_args()


def download_annotations(split: str, cache_dir: Path) -> Path:
    split_dir = SPLIT_DIRS[split]
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"{split_dir}/_annotations.coco.json",
            cache_dir=cache_dir,
        )
    )


def load_coco(annotation_path: Path) -> dict[str, Any]:
    return json.loads(annotation_path.read_text(encoding="utf-8"))


def select_images(coco: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        annotations_by_image[int(ann["image_id"])].append(ann)

    images_with_boxes = [
        image for image in coco.get("images", [])
        if annotations_by_image.get(int(image["id"]))
    ]
    rng = random.Random(seed)
    rng.shuffle(images_with_boxes)
    return images_with_boxes[:count]


def category_mapping(coco: dict[str, Any]) -> tuple[dict[int, int], list[str]]:
    categories = sorted(coco.get("categories", []), key=lambda item: int(item["id"]))
    category_id_to_yolo = {int(category["id"]): idx for idx, category in enumerate(categories)}
    names = [str(category["name"]) for category in categories]
    return category_id_to_yolo, names


def download_image(split: str, file_name: str, cache_dir: Path) -> Path:
    split_dir = SPLIT_DIRS[split]
    normalized = file_name.replace("\\", "/")
    if normalized.startswith(split_dir):
        hf_path = normalized
    else:
        hf_path = f"{split_dir}/{normalized}"
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=hf_path,
            cache_dir=cache_dir,
        )
    )


def write_yolo_label(
    output_path: Path,
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    category_id_to_yolo: dict[int, int],
) -> None:
    width = float(image["width"])
    height = float(image["height"])
    rows = []
    for ann in annotations:
        x, y, box_w, box_h = [float(value) for value in ann["bbox"]]
        if box_w <= 0 or box_h <= 0:
            continue
        x_center = (x + box_w / 2.0) / width
        y_center = (y + box_h / 2.0) / height
        norm_w = box_w / width
        norm_h = box_h / height
        values = [
            category_id_to_yolo[int(ann["category_id"])],
            _clamp01(x_center),
            _clamp01(y_center),
            _clamp01(norm_w),
            _clamp01(norm_h),
        ]
        rows.append(" ".join([str(values[0])] + [f"{value:.6f}" for value in values[1:]]))
    output_path.write_text("\n".join(rows), encoding="utf-8")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def convert_split(
    *,
    split: str,
    count: int,
    output_root: Path,
    cache_dir: Path,
    seed: int,
) -> list[str]:
    annotation_path = download_annotations(split, cache_dir)
    coco = load_coco(annotation_path)
    selected_images = select_images(coco, count, seed)
    category_id_to_yolo, names = category_mapping(coco)

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        annotations_by_image[int(ann["image_id"])].append(ann)

    image_output_dir = output_root / "images" / split
    label_output_dir = output_root / "labels" / split
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for image in tqdm(selected_images, desc=f"Preparing {split}"):
        source_image = download_image(split, str(image["file_name"]), cache_dir)
        target_image = image_output_dir / Path(str(image["file_name"])).name
        shutil.copy2(source_image, target_image)
        label_path = label_output_dir / f"{target_image.stem}.txt"
        write_yolo_label(
            label_path,
            image,
            annotations_by_image[int(image["id"])],
            category_id_to_yolo,
        )

    return names


def write_data_yaml(output_root: Path, names: list[str]) -> Path:
    data = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(names)},
    }
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    cache_dir = Path(args.cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_names = convert_split(
        split="train",
        count=args.train_count,
        output_root=output_root,
        cache_dir=cache_dir,
        seed=args.seed,
    )
    val_names = convert_split(
        split="val",
        count=args.val_count,
        output_root=output_root,
        cache_dir=cache_dir,
        seed=args.seed + 1,
    )
    names = train_names or val_names
    yaml_path = write_data_yaml(output_root, names)
    print(f"YOLO dataset written to: {output_root.resolve()}")
    print(f"Data YAML: {yaml_path.resolve()}")
    print(f"Classes: {', '.join(names)}")


if __name__ == "__main__":
    main()

