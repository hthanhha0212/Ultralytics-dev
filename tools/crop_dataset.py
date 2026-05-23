import argparse
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop YOLO-format boxes for a class id into a new dataset.")
    parser.add_argument("--images", required=True, help="Path to image folder.")
    parser.add_argument("--labels", required=True, help="Path to label folder.")
    parser.add_argument("--class-id", type=int, required=True, help="Class id to crop.")
    parser.add_argument("--output", required=True, help="Output folder for crops.")
    parser.add_argument(
        "--suffix",
        default="crop",
        help="Suffix for output filenames (default: crop).",
    )
    parser.add_argument(
        "--output-labels",
        required=True,
        help="Output folder for updated labels.",
    )
    parser.add_argument(
        "--class-id-output",
        type=int,
        default=1,
        help="Class id to use in output labels (default: 1).",
    )
    parser.add_argument(
        "--class-id-license",
        type=int,
        default=1,
        help="License plate class id in input labels (default: 1).",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[416, 416],
        help="Target size for letterbox resize (default: 416 416).",
    )
    parser.add_argument(
        "--pad",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=[114, 114, 114],
        help="Padding color for letterbox (default: 114 114 114).",
    )
    return parser.parse_args()


def clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))


def iter_images(image_dir: Path):
    for path in image_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            yield path


def letterbox(
    image: Image.Image, target_size: tuple[int, int], pad_color: tuple[int, int, int]
) -> tuple[Image.Image, float, int, int]:
    target_w, target_h = target_size
    src_w, src_h = image.size
    if src_w == 0 or src_h == 0:
        return Image.new("RGB", (target_w, target_h), pad_color), 1.0, 0, 0

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def yolo_to_xyxy(
    xc: float, yc: float, bw: float, bh: float, width: int, height: int
) -> tuple[float, float, float, float]:
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return x1, y1, x2, y2


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    if width <= 0 or height <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return xc / width, yc / height, bw / width, bh / height


def main() -> int:
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    output_dir = Path(args.output)
    output_label_dir = Path(args.output_labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    class_id = args.class_id
    class_id_license = args.class_id_license
    class_id_output = args.class_id_output
    suffix = args.suffix
    target_size = (args.size[0], args.size[1])
    pad_color = (args.pad[0], args.pad[1], args.pad[2])

    for image_path in iter_images(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            continue

        width, height = image.size
        with label_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        crop_index = 0
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                label_class = int(float(parts[0]))
                if label_class != class_id:
                    continue
                xc, yc, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, width, height)

            x1 = clamp(x1, 0, width)
            y1 = clamp(y1, 0, height)
            x2 = clamp(x2, 0, width)
            y2 = clamp(y2, 0, height)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2))
            crop_w, crop_h = crop.size
            if crop_w == 0 or crop_h == 0:
                continue

            license_boxes = []
            for lic_line in lines:
                lic_parts = lic_line.split()
                if len(lic_parts) < 5:
                    continue
                try:
                    lic_class = int(float(lic_parts[0]))
                    if lic_class != class_id_license:
                        continue
                    lxc, lyc, lbw, lbh = map(float, lic_parts[1:5])
                except ValueError:
                    continue

                lx1, ly1, lx2, ly2 = yolo_to_xyxy(lxc, lyc, lbw, lbh, width, height)
                lcx = (lx1 + lx2) / 2.0
                lcy = (ly1 + ly2) / 2.0
                if lcx < x1 or lcx > x2 or lcy < y1 or lcy > y2:
                    continue

                lx1 = clamp(lx1 - x1, 0, crop_w)
                ly1 = clamp(ly1 - y1, 0, crop_h)
                lx2 = clamp(lx2 - x1, 0, crop_w)
                ly2 = clamp(ly2 - y1, 0, crop_h)
                if lx2 <= lx1 or ly2 <= ly1:
                    continue
                license_boxes.append((lx1, ly1, lx2, ly2))

            crop, scale, pad_x, pad_y = letterbox(crop, target_size, pad_color)
            crop_index += 1
            output_name = f"{image_path.stem}_{suffix}_{crop_index}{image_path.suffix}"
            crop.save(output_dir / output_name)

            label_name = f"{image_path.stem}_{suffix}_{crop_index}.txt"
            label_path = output_label_dir / label_name
            with label_path.open("w", encoding="utf-8") as label_handle:
                for lx1, ly1, lx2, ly2 in license_boxes:
                    lx1 = lx1 * scale + pad_x
                    ly1 = ly1 * scale + pad_y
                    lx2 = lx2 * scale + pad_x
                    ly2 = ly2 * scale + pad_y
                    lx1 = clamp(lx1, 0, target_size[0])
                    ly1 = clamp(ly1, 0, target_size[1])
                    lx2 = clamp(lx2, 0, target_size[0])
                    ly2 = clamp(ly2, 0, target_size[1])
                    if lx2 <= lx1 or ly2 <= ly1:
                        continue
                    nxc, nyc, nbw, nbh = xyxy_to_yolo(lx1, ly1, lx2, ly2, target_size[0], target_size[1])
                    label_handle.write(f"{class_id_output} {nxc:.6f} {nyc:.6f} {nbw:.6f} {nbh:.6f}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
