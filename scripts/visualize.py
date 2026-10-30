#!/usr/bin/env python3
"""Run a trained checkpoint on a folder of images and visualize the detections.

Predictions are drawn in magenta with their score. If the images sit in an Ultralytics-style
.../images/ dir with a sibling .../labels/ dir, ground truth is drawn in green and recall /
precision at IoU 0.5 are printed.

Writes annotated images and 6x6 contact sheets to --out. With --show, opens a viewer:
    right arrow / space / d: next     left arrow / a: previous     q / Esc: quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PACKAGE_DIR = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
GT_COLOR, PRED_COLOR = (0, 255, 0), (255, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=PACKAGE_DIR / "runs/yolo26n/weights/best.pt")
    parser.add_argument("--source", type=Path, default=PACKAGE_DIR / "data/merged/test/images",
                        help="Folder of images.")
    parser.add_argument("--out", type=Path, default=PACKAGE_DIR / "runs/visualize")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help='"cpu" keeps the GPU free for training.')
    parser.add_argument("--show", action="store_true", help="Step through results in a window.")
    return parser.parse_args()


def load_gt(image: Path, w: int, h: int) -> np.ndarray | None:
    """GT boxes as xyxy pixels, or None if the image has no label file."""
    label = Path(str(image.parent).replace("/images", "/labels")) / f"{image.stem}.txt"
    if "/images" not in str(image.parent) or not label.exists():
        return None
    rows = np.loadtxt(label, ndmin=2).reshape(-1, 5)[:, 1:] * [w, h, w, h]
    return np.hstack((rows[:, :2] - rows[:, 2:] / 2, rows[:, :2] + rows[:, 2:] / 2))


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, shape (len(a), len(b))."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def draw(img: np.ndarray, gt: np.ndarray | None, pred: np.ndarray, conf: np.ndarray) -> None:
    t = max(2, round(max(img.shape[:2]) / 300))
    for x1, y1, x2, y2 in ([] if gt is None else gt.astype(int)):
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, t)
    for (x1, y1, x2, y2), c in zip(pred.astype(int), conf):
        cv2.rectangle(img, (x1, y1), (x2, y2), PRED_COLOR, t)
        cv2.putText(img, f"{c:.2f}", (x1, max(y1 - 4, 14 * t)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * t, PRED_COLOR, t)


def tile(img: np.ndarray, size: int = 320) -> np.ndarray:
    s = size / max(img.shape[:2])
    small = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)))
    out = np.full((size, size, 3), 40, np.uint8)
    out[:small.shape[0], :small.shape[1]] = small
    return out


def show(images: list[tuple[str, np.ndarray]]) -> None:
    i = 0
    cv2.namedWindow("cones", cv2.WINDOW_NORMAL)
    while True:
        name, img = images[i]
        cv2.setWindowTitle("cones", f"[{i + 1}/{len(images)}] {name}")
        cv2.imshow("cones", img)
        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27) or cv2.getWindowProperty("cones", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("a"), 65361, 2424832):  # left arrow (X11 / Windows)
            i = (i - 1) % len(images)
        else:
            i = (i + 1) % len(images)
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    files = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    results, n_gt, n_pred, n_hit, has_gt = [], 0, 0, 0, False
    for f in files:
        img = cv2.imread(str(f))
        r = model.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device,
                          verbose=False)[0]
        pred, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        gt = load_gt(f, img.shape[1], img.shape[0])
        n_pred += len(pred)
        if gt is not None:
            has_gt = True
            n_gt += len(gt)
            if len(gt) and len(pred):
                n_hit += int((iou(gt, pred).max(1) > 0.5).sum())
        draw(img, gt, pred, conf)
        cv2.imwrite(str(args.out / f.name), img)
        results.append((f.name, img))

    tiles = [tile(img) for _, img in results]
    blank = np.full_like(tiles[0], 40)
    for k in range(0, len(tiles), 36):
        page = tiles[k:k + 36] + [blank] * (36 - len(tiles[k:k + 36]))
        sheet = np.vstack([np.hstack(page[i:i + 6]) for i in range(0, 36, 6)])
        cv2.imwrite(str(args.out / f"sheet_{k // 36}.jpg"), sheet)

    print(f"{len(files)} images, {n_pred} predictions @ conf {args.conf} -> {args.out}")
    if has_gt:
        print(f"{n_gt} GT cones, {n_hit} matched (IoU > 0.5): recall {n_hit / max(n_gt, 1):.2f}, "
              f"precision {n_hit / max(n_pred, 1):.2f}")
    if args.show:
        show(results)


if __name__ == "__main__":
    main()
