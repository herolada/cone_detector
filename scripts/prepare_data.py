#!/usr/bin/env python3
"""Merge the raw cone datasets in data/ into one Ultralytics detection dataset.

- Polygon labels (some Roboflow exports have them) are converted to boxes; classes become 0.
- TraCon contains cone-free road images with empty label files; they are kept as negatives.
- Byte-identical images are deduplicated; a duplicate is kept in the evaluation split (test, then
  val) and dropped from train, so no image leaks from train into evaluation.

Output: data/merged/{train,val,test}/{images,labels} and data/merged/data.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# name -> (dataset root, {merged split: (image dir, label dir) relative to root})
SOURCES = {
    "roboflow": (
        DATA_DIR / "Traffic Cones.v1-default-version.yolov8",
        {s: (f"{d}/images", f"{d}/labels") for s, d in
         (("train", "train"), ("val", "valid"), ("test", "test"))},
    ),
    "tracon": (
        DATA_DIR / "Cone-Detection" / "TraCon_dataset",
        {s: (f"images/{s}", f"labels/{s}") for s in ("train", "val", "test")},
    ),
}
SPLITS = ("test", "val", "train")  # evaluation splits first, so they win deduplication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "merged", help="Output directory.")
    parser.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES),
                        help="Which raw datasets to include.")
    parser.add_argument("--min-size", type=float, default=0.002,
                        help="Drop boxes narrower or shorter than this (normalized).")
    return parser.parse_args()


def convert_label(path: Path, min_size: float) -> list[str]:
    """Return YOLO box lines (class forced to 0) from a box or polygon label file."""
    lines = []
    for line in path.read_text().splitlines():
        values = [float(v) for v in line.split()[1:]]
        if len(values) == 4:
            cx, cy, w, h = values
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif len(values) >= 6 and len(values) % 2 == 0:  # polygon x1 y1 x2 y2 ...
            xs, ys = values[0::2], values[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, 1.0), min(y2, 1.0)
        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue
        lines.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return lines


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)

    seen: set[str] = set()
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        for name in args.sources:
            root, split_dirs = SOURCES[name]
            image_dir, label_dir = (root / d for d in split_dirs[split])
            n_images = n_boxes = n_dupes = 0
            for image in sorted(image_dir.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                digest = hashlib.md5(image.read_bytes()).hexdigest()
                if digest in seen:
                    n_dupes += 1
                    continue
                seen.add(digest)

                label = label_dir / f"{image.stem}.txt"
                lines = convert_label(label, args.min_size) if label.exists() else []
                stem = f"{name}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', image.stem)}"
                link_or_copy(image, out / split / "images" / f"{stem}{image.suffix.lower()}")
                (out / split / "labels" / f"{stem}.txt").write_text(
                    "".join(f"{l}\n" for l in lines))
                n_images += 1
                n_boxes += len(lines)
            print(f"{split:5s} {name:8s}: {n_images:4d} images, {n_boxes:4d} boxes, "
                  f"{n_dupes} duplicates skipped")

    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 1\nnames: ['cone']\n")
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
