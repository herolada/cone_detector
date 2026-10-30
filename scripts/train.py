#!/usr/bin/env python3
"""Fine-tune a COCO-pretrained Ultralytics YOLO on the merged cone dataset and export to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from export import export_onnx

PACKAGE_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26n.pt",
                        help="Pretrained checkpoint, e.g. yolo26n.pt, yolo11n.pt, yolo26s.pt.")
    parser.add_argument("--data", type=Path, default=PACKAGE_DIR / "data" / "merged" / "data.yaml",
                        help="Dataset YAML (created by prepare_data.py).")
    parser.add_argument("--imgsz", type=int, default=640, help="Training and export image size.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16, help="16 fits a 4 GB GPU at 640 px.")
    parser.add_argument("--device", default="0", help='"0", "0,1" or "cpu".')
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience.")
    parser.add_argument("--project", type=Path, default=PACKAGE_DIR / "runs")
    parser.add_argument("--name", default=None, help="Run name, defaults to the model stem.")
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX export to models/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.name or Path(args.model).stem,
        exist_ok=True,
        cos_lr=True,
        # Cones are defined by their orange color, so keep hue jitter small; brightness and
        # saturation jitter cover lighting changes.
        hsv_h=0.01,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=5.0,
        scale=0.6,  # wide zoom range: cones are seen from close up to far away
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
    )

    best = Path(model.trainer.best)
    print(f"Best checkpoint: {best}")
    YOLO(str(best)).val(data=str(args.data.resolve()), imgsz=args.imgsz, split="test",
                        device=args.device, project=str(args.project.resolve()),
                        name=f"{best.parent.parent.name}_test", exist_ok=True)
    if not args.no_export:
        export_onnx(best, args.imgsz)


if __name__ == "__main__":
    main()
