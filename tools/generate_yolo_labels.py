#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield path
    else:
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield path


def find_white_bbox(image_path: Path, threshold: int) -> tuple[int, int, int, int] | None:
    img = Image.open(image_path).convert("L")
    bw = img.point(lambda p: 255 if p >= threshold else 0)
    return bw.getbbox()


def bbox_to_yolo(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    left, top, right, bottom = bbox
    x_center = (left + right) / 2.0 / width
    y_center = (top + bottom) / 2.0 / height
    w = (right - left) / width
    h = (bottom - top) / height
    return x_center, y_center, w, h


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate YOLO labels from white-rectangle-on-black images.")
    parser.add_argument("--input-dir", required=True, help="Directory with input images.")
    parser.add_argument("--output-dir", required=True, help="Directory to write .txt label files.")
    parser.add_argument("--class-id", type=int, default=0, help="Class id to use in labels.")
    parser.add_argument("--threshold", type=int, default=200, help="White threshold (0-255).")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    parser.add_argument("--empty-ok", action="store_true", help="Write empty label files when no bbox is found.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in iter_images(input_dir, args.recursive):
        bbox = find_white_bbox(img_path, args.threshold)
        label_path = output_dir / (img_path.stem + ".txt")

        if bbox is None:
            if args.empty_ok:
                label_path.write_text("")
            continue

        with Image.open(img_path) as img:
            width, height = img.size

        x_center, y_center, w, h = bbox_to_yolo(bbox, width, height)
        line = f"{args.class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
        label_path.write_text(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
