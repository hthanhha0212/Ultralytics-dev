"""
GPU-oriented loss analysis tool for Ultralytics YOLO detection models.

The script lives inside the Ultralytics repository but does not modify any library internals.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, TQDM


@dataclass
class LossSample:
    image: str
    total: float
    components: list[float]


def _format_names(names) -> dict[int, str]:
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    raise TypeError("names must be dict or list.")


def resolve_dataset_cfg(data, images, default_names, channels):
    if data is None:
        if images is None:
            raise ValueError("Provide either --data YAML or --images directory.")
        names = _format_names(default_names)
        return {"train": images, "val": images, "names": names, "nc": len(names), "channels": channels}

    if isinstance(data, (str, Path)):
        cfg = check_det_dataset(str(data))
    elif isinstance(data, dict):
        cfg = data.copy()
    else:
        raise TypeError("--data must be path or dict.")

    cfg["names"] = _format_names(cfg.get("names", default_names))
    cfg.setdefault("nc", len(cfg["names"]))
    cfg.setdefault("channels", channels)
    return cfg


def _prepare_batch(batch, device, half: bool):
    imgs = batch["img"].to(device, non_blocking=True)
    imgs = imgs.half() if half else imgs.float()
    batch["img"] = imgs / 255
    for key in ("cls", "bboxes", "batch_idx"):
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)


def _single_from_batch(batch: dict, idx: int) -> dict:
    mask = batch["batch_idx"] == idx
    single = {
        "img": batch["img"][idx : idx + 1],
        "cls": batch["cls"][mask],
        "bboxes": batch["bboxes"][mask],
        "batch_idx": torch.zeros_like(batch["batch_idx"][mask]),
    }
    if "im_file" in batch:
        single["im_file"] = [batch["im_file"][idx]]
    return single


def compute_losses(
    model,
    data_cfg: dict,
    split: str,
    imgsz: int | Iterable[int],
    batch: int,
    workers: int,
    half: bool,
    max_samples: int | None,
    show_progress: bool,
) -> list[LossSample]:
    split_path = data_cfg.get(split)
    if split_path is None:
        raise KeyError(f"Split '{split}' missing in dataset config.")

    stride = int(getattr(model, "stride", torch.tensor([32])).max())
    cfg = get_cfg(DEFAULT_CFG_DICT, overrides={"imgsz": imgsz, "batch": batch, "rect": True})
    cfg.task = "detect"
    cfg.data = data_cfg
    cfg.workers = workers
    dataset = build_yolo_dataset(cfg, img_path=split_path, batch=batch, data=data_cfg, mode="val", rect=True, stride=stride)
    dataloader = build_dataloader(dataset, batch=batch, workers=workers, shuffle=False, rank=-1, drop_last=False)

    iterator = TQDM(dataloader, total=len(dataset), desc="Loss", unit="img") if show_progress else dataloader
    samples: list[LossSample] = []
    was_training = model.training
    model.eval()
    processed = 0

    with torch.inference_mode():
        for batch_dict in iterator:
            _prepare_batch(batch_dict, next(model.parameters()).device, half)
            for idx in range(batch_dict["img"].shape[0]):
                single = _single_from_batch(batch_dict, idx)
                loss_tensor, components = model(single)
                comp_vals = [float(v) for v in components.tolist()]
                img_path = single.get("im_file", ["unknown"])[0]
                samples.append(LossSample(image=img_path, total=float(loss_tensor.sum().item()), components=comp_vals))
                processed += 1
                if max_samples and processed >= max_samples:
                    break
            if max_samples and processed >= max_samples:
                break

    if was_training:
        model.train()
    return samples


def analyze(args):
    model = YOLO(args.model)
    if model.task != "detect":
        raise NotImplementedError("Only detection checkpoints supported.")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if hasattr(model.model, "criterion"):
        model.model.criterion = None
    if hasattr(model.model, "args") and model.model.args is not None:
        arg_data = model.model.args
        overrides = arg_data if isinstance(arg_data, dict) else vars(arg_data)
        model.model.args = get_cfg(DEFAULT_CFG_DICT, overrides=overrides)

    names = getattr(model.model, "names", model.names)
    channels = getattr(getattr(model.model, "model", None), "args", {}).get("channels", 3)
    data_cfg = resolve_dataset_cfg(args.data, args.images, names, channels)

    use_half = bool(args.half and model.device.type == "cuda")
    samples = compute_losses(
        model.model,
        data_cfg=data_cfg,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        half=use_half,
        max_samples=args.max_samples,
        show_progress=not args.quiet,
    )
    if not samples:
        LOGGER.warning("No samples processed.")
        return []

    ordered = sorted(samples, key=lambda s: s.total, reverse=True)
    topn = min(args.topk, len(ordered))
    LOGGER.info(f"Processed {len(samples)} images from '{args.split}'. Showing top {topn}.")
    for sample in ordered[:topn]:
        LOGGER.info(
            f"{sample.total:.4f} | box={sample.components[0]:.4f}, cls={sample.components[1]:.4f}, "
            f"dfl={sample.components[2]:.4f} :: {sample.image}"
        )

    if args.save_csv:
        csv_path = Path(args.save_csv).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "total_loss", "box_loss", "cls_loss", "dfl_loss"])
            for sample in ordered:
                writer.writerow([sample.image, sample.total, *sample.components])
        LOGGER.info(f"Saved CSV to {csv_path}")

    return ordered


def parse_args():
    parser = argparse.ArgumentParser(description="GPU loss ranking for Ultralytics YOLO models.")
    parser.add_argument("--model", required=True, help="Path to YOLO checkpoint (.pt).")
    parser.add_argument("--data", help="Dataset YAML path.")
    parser.add_argument("--images", help="Image directory when YAML unavailable.")
    parser.add_argument("--split", default="train", help="Dataset split key.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--topk", type=int, default=20, help="Number of hardest images to log.")
    parser.add_argument("--max-samples", type=int, help="Optional cap on processed images.")
    parser.add_argument("--save-csv", help="Optional CSV output path.")
    parser.add_argument("--half", action="store_true", help="Enable FP16 when supported.")
    parser.add_argument("--device", help="Torch device string. Defaults to CUDA if available.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
