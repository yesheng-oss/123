from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .dataset import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def save_labelme_json(
    out_path: Path,
    image_name: str,
    image_width: int,
    image_height: int,
    detections: list[Detection],
) -> None:
    shapes = []
    for detection in detections:
        shapes.append(
            {
                "label": detection.label,
                "points": [
                    [float(detection.x1), float(detection.y1)],
                    [float(detection.x2), float(detection.y2)],
                ],
                "group_id": None,
                "description": None,
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
                "score": float(detection.confidence),
            }
        )

    payload = {
        "version": "5.8.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_prediction(
    weights: Path,
    input_path: Path,
    output: Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | None = None,
    batch_size: int = 1,
) -> None:
    import cv2
    from ultralytics import YOLO

    output_images = output / "images"
    output_json = output / "json"
    output_images.mkdir(parents=True, exist_ok=True)
    output_json.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    image_paths = list(_iter_images(input_path))
    csv_rows: list[dict[str, object]] = []

    for batch in _batched(image_paths, batch_size):
        results = model.predict(
            source=[str(path) for path in batch],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )

        for image_path, result in zip(batch, results):
            image = result.orig_img.copy() if getattr(result, "orig_img", None) is not None else cv2.imread(str(image_path))
            if image is None:
                print(f"warning: could not read image {image_path}")
                continue

            height, width = image.shape[:2]
            detections: list[Detection] = []
            for box in result.boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = result.names.get(class_id, str(class_id))
                detection = Detection(label, confidence, x1, y1, x2, y2)
                detections.append(detection)
                _draw_detection(cv2, image, detection)
                csv_rows.append(
                    {
                        "image": image_path.name,
                        "label": label,
                        "confidence": f"{confidence:.6f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                    }
                )

            cv2.imwrite(str(output_images / image_path.name), image)
            save_labelme_json(output_json / f"{image_path.stem}.json", image_path.name, width, height, detections)

    _write_predictions_csv(output / "predictions.csv", csv_rows)
    print(f"Annotated images: {output_images}")
    print(f"LabelMe JSON: {output_json}")
    print(f"CSV summary: {output / 'predictions.csv'}")


def _iter_images(input_path: Path):
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield input_path
        return
    for path in sorted(input_path.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _batched(items, batch_size: int):
    batch_size = max(1, int(batch_size))
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _draw_detection(cv2, image, detection: Detection) -> None:
    color = (0, 0, 180) if detection.label == "th" else (0, 150, 0)
    start = (int(round(detection.x1)), int(round(detection.y1)))
    end = (int(round(detection.x2)), int(round(detection.y2)))
    cv2.rectangle(image, start, end, color, 2)
    text = f"{detection.label} {detection.confidence:.2f}"
    cv2.putText(image, text, (start[0], max(15, start[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def _write_predictions_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "label", "confidence", "x1", "y1", "x2", "y2"]
    with out_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect answer-card boxes and export annotations.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    run_prediction(args.weights, args.input, args.output, args.imgsz, args.conf, args.device, args.batch_size)


if __name__ == "__main__":
    main()
