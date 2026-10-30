#!/usr/bin/env python3
"""Export a trained checkpoint to a static-shape ONNX model in the ROS package's models/ dir.

The ONNX model is portable: copy it to the Jetson and build a TensorRT engine there (engines are
tied to the GPU and TensorRT version they were built with):
    /usr/src/tensorrt/bin/trtexec --onnx=cone_yolo.onnx --saveEngine=cone_yolo.engine --fp16
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def export_onnx(checkpoint: Path, imgsz: int, output: Path = MODELS_DIR / "cone_yolo.onnx",
                opset: int = 17) -> Path:
    exported = YOLO(str(checkpoint)).export(format="onnx", imgsz=imgsz, opset=opset,
                                            simplify=True, dynamic=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output)
    print(f"ONNX model: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Trained .pt, e.g. runs/yolo26n/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "cone_yolo.onnx")
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.imgsz, args.output, args.opset)


if __name__ == "__main__":
    main()
