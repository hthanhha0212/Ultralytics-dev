# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics import YOLO


def prepare_qat_model(
    model_config: str | Path,
    weights: str | Path | None = None,
    *,
    qconfig: torch.ao.quantization.QConfig | None = None,
    prepare_inplace: bool = True,
    show_info: bool = False,
) -> YOLO:
    """Build a YOLO model configured for Quantization Aware Training (QAT). Note that only YOLOv10 model works for this
    setup.

    Args:
        model_config: Path to a YOLO model configuration file (e.g. ``qyolov10n.yaml``).
        weights: Optional path to pretrained weights to load before preparing for QAT.
        qconfig: Optional explicit QAT configuration. Defaults to ``get_default_qat_qconfig``.
        prepare_inplace: Whether to run ``torch.ao.quantization.prepare_qat`` inplace.
        show_info: If ``True``, call ``model.info()`` before returning the prepared model.

    Returns:
        The configured YOLO model ready for QAT.
    """
    model = YOLO(str(model_config), do_qat=True)

    if weights:
        model.load(str(weights))

    # Fuse convolution, batch-norm, and activation blocks prior to quantization.
    model.fuse()

    if qconfig is None:
        engine = torch.backends.quantized.engine
        if engine == "none":
            engine = "fbgemm"
        get_qconfig = getattr(torch.ao.quantization, "get_default_qat_qconfig", None)
        if get_qconfig is None:
            get_qconfig = torch.quantization.get_default_qat_qconfig
        qconfig = get_qconfig(engine)

    # Apply the QAT configuration to the full model.
    model.model.qconfig = qconfig

    # Automatically disable the DFL head for YOLOv10.
    try:
        head = model.model.model[23]
    except (AttributeError, IndexError) as exc:  # pragma: no cover - defensive safeguard
        raise RuntimeError("Failed to locate DFL head at model.model.model[23] for YOLOv10.") from exc
    if hasattr(head, "qconfig"):
        head.qconfig = None
    dfl = getattr(head, "dfl", None)
    if dfl is not None and hasattr(dfl, "qconfig"):
        dfl.qconfig = None

    prepare_qat = getattr(torch.ao.quantization, "prepare_qat", None)
    if prepare_qat is None:
        prepare_qat = torch.quantization.prepare_qat
    prepare_qat(model.model, inplace=prepare_inplace)

    if show_info:
        model.info()

    return model


def run_post_training_quantization(
    weights: str | Path,
    data: str | Path,
    *,
    imgsz: int = 640,
    batch: int = 16,
    convert_inplace: bool = True,
    calibrate: bool = True,
    verbose: bool = True,
    val_kwargs: dict | None = None,
) -> YOLO:
    model = YOLO(str(weights), q=True)
    model.model.eval()

    model.model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

    model.model.model[23].dfl.qconfig = None

    torch.ao.quantization.prepare(model.model, inplace=True)

    if calibrate:
        if verbose:
            print("Performing PTQ calibration...")
        val_args: dict = {"data": str(data), "imgsz": imgsz, "batch": batch}
        if val_kwargs:
            val_args.update(val_kwargs)
        model.val(**val_args)
        if verbose:
            print("Calibration done.")

    torch.ao.quantization.convert(model.model, inplace=True)
    model.model.eval()

    return model


def load_ptq_model_from_state_dict(
    base_weights: str | Path,
    quant_state_dict: str | Path,
) -> YOLO:
    """Rebuild a quantized YOLOv10 model shell and load pre-computed quantized weights.

    Args:
        base_weights: Path to the floating-point checkpoint used to instantiate the shell.
        quant_state_dict: Path to the saved quantized ``state_dict``.
        map_location: Device mapping passed to ``torch.load`` (defaults to CPU).
        strict: Forwarded to ``load_state_dict``.

    Returns:
        The restored quantized ``YOLO`` model instance.
    """
    model = YOLO(str(base_weights), q=True)
    model.model.eval()

    model.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    model.model.model[23].dfl.qconfig = None

    torch.quantization.prepare(model.model, inplace=True)

    model.fuse()

    torch.quantization.convert(model.model, inplace=True)

    model.model.load_state_dict(torch.load(str(quant_state_dict)))

    model.model.eval()

    return model


def load_qat_model_from_state_dict(
    base_weights: str | Path,
    quant_state_dict: str | Path,
) -> YOLO:
    """Rebuild a quantized YOLOv10 model shell and load pre-computed quantized weights.

    Args:
        base_weights: Path to the floating-point checkpoint used to instantiate the shell.
        quant_state_dict: Path to the saved quantized ``state_dict``.
        map_location: Device mapping passed to ``torch.load`` (defaults to CPU).
        strict: Forwarded to ``load_state_dict``.

    Returns:
        The restored quantized ``YOLO`` model instance.
    """
    model = YOLO(str(base_weights), do_qat=True)
    model.fuse()

    model.model.qconfig = torch.ao.quantization.get_default_qat_qconfig()

    model.model.model[23].dfl.qconfig = None

    model.model.train()

    torch.quantization.prepare_qat(model.model, inplace=True)

    torch.quantization.convert(model.model, inplace=True)

    model.model.model[23].fuse()

    model.model.load_state_dict(torch.load(str(quant_state_dict)))

    model.model.eval()

    return model


def get_lib_quant_model() -> YOLO:
    """Load the library-provided quantized model using checkpoints stored relative to this package."""
    package_root = Path(__file__).resolve().parent.parent
    base_weights = package_root / "pretrained" / "weights" / "best.pt"
    quant_state_dict = package_root / "quant" / "quant_state_dict" / "qat_sttd.pt"

    if not base_weights.exists():
        raise FileNotFoundError(f"Base weights not found at {base_weights}")
    if not quant_state_dict.exists():
        raise FileNotFoundError(f"Quantized state dict not found at {quant_state_dict}")

    return load_qat_model_from_state_dict(
        base_weights=base_weights,
        quant_state_dict=quant_state_dict,
    )
