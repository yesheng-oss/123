from __future__ import annotations

import argparse
from pathlib import Path

from answer_card.dataset import build_yolo_dataset


def choose_device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO model for answer-card detection.")
    parser.add_argument("--source", type=Path, default=Path(r"C:\Users\22234\Desktop\kgt\训练集"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/answer_card_yolo"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_yolo_dataset(args.source, args.dataset, args.val_ratio, args.seed)
    print(f"Dataset ready: {summary.dataset_dir}")
    print(f"train={summary.train_count}, val={summary.val_count}, warnings={summary.warning_count}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(summary.data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=choose_device(args.device),
        project=str(Path("runs/detect").resolve()),
        name="answer_card",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
