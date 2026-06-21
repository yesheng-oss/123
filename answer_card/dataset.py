from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import CLASSES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class DatasetSummary:
    dataset_dir: Path
    data_yaml: Path
    train_count: int
    val_count: int
    skipped_count: int
    warning_count: int


def convert_labelme_to_yolo(data: dict) -> tuple[list[str], list[str]]:
    width = float(data.get("imageWidth") or 0)
    height = float(data.get("imageHeight") or 0)
    warnings: list[str] = []
    rows: list[str] = []

    if width <= 0 or height <= 0:
        return rows, ["invalid imageWidth/imageHeight"]

    for index, shape in enumerate(data.get("shapes", [])):
        label = shape.get("label")
        if label not in CLASS_TO_ID:
            continue
        if shape.get("shape_type") != "rectangle":
            warnings.append(f"shape {index}: unsupported shape_type {shape.get('shape_type')!r}")
            continue
        points = shape.get("points") or []
        if len(points) != 2:
            warnings.append(f"shape {index}: rectangle must have exactly 2 points")
            continue

        try:
            x1, y1 = map(float, points[0])
            x2, y2 = map(float, points[1])
        except (TypeError, ValueError):
            warnings.append(f"shape {index}: invalid point values")
            continue

        x1 = _clamp(x1, 0.0, width)
        x2 = _clamp(x2, 0.0, width)
        y1 = _clamp(y1, 0.0, height)
        y2 = _clamp(y2, 0.0, height)
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        box_width = right - left
        box_height = bottom - top
        if box_width <= 0 or box_height <= 0:
            warnings.append(f"shape {index}: empty rectangle")
            continue

        x_center = (left + right) / 2.0 / width
        y_center = (top + bottom) / 2.0 / height
        norm_width = box_width / width
        norm_height = box_height / height
        rows.append(
            f"{CLASS_TO_ID[label]} {x_center:.6f} {y_center:.6f} "
            f"{norm_width:.6f} {norm_height:.6f}"
        )

    return rows, warnings


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_yolo_dataset(
    source: Path,
    output: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetSummary:
    source = source.resolve()
    output = output.resolve()
    pairs, skipped = _find_image_json_pairs(source)
    random.Random(seed).shuffle(pairs)

    val_count = int(round(len(pairs) * val_ratio))
    if len(pairs) > 1:
        val_count = min(max(1, val_count), len(pairs) - 1)
    train_pairs = pairs[val_count:]
    val_pairs = pairs[:val_count]

    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    warning_count = 0
    train_written = _write_split(train_pairs, output, "train")
    val_written = _write_split(val_pairs, output, "val")
    warning_count += train_written[1] + val_written[1]

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: th",
                "  1: option",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return DatasetSummary(
        dataset_dir=output,
        data_yaml=data_yaml,
        train_count=train_written[0],
        val_count=val_written[0],
        skipped_count=skipped,
        warning_count=warning_count,
    )


def _find_image_json_pairs(source: Path) -> tuple[list[tuple[Path, Path]], int]:
    pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for image_path in sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            pairs.append((image_path, json_path))
        else:
            skipped += 1
    return pairs, skipped


def _write_split(pairs: list[tuple[Path, Path]], output: Path, split: str) -> tuple[int, int]:
    written = 0
    warning_count = 0
    for image_path, json_path in pairs:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        rows, warnings = convert_labelme_to_yolo(data)
        warning_count += len(warnings)
        if not rows:
            warning_count += 1
            continue

        shutil.copy2(image_path, output / "images" / split / image_path.name)
        label_path = output / "labels" / split / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        written += 1
    return written, warning_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON files to a YOLO dataset.")
    parser.add_argument("--source", type=Path, default=Path(r"C:\Users\22234\Desktop\kgt\训练集"))
    parser.add_argument("--output", type=Path, default=Path("datasets/answer_card_yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_yolo_dataset(args.source, args.output, args.val_ratio, args.seed)
    print(f"Dataset written to: {summary.dataset_dir}")
    print(f"data.yaml: {summary.data_yaml}")
    print(f"train images: {summary.train_count}")
    print(f"val images: {summary.val_count}")
    print(f"skipped images without JSON: {summary.skipped_count}")
    print(f"warnings: {summary.warning_count}")


if __name__ == "__main__":
    main()
