# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import namedtuple

import torch
from torch.ao.nn.quantized.modules.conv import Conv2d as QConv2d

__all__ = (
    "QuantParams",
    "compare_list",
    "compare_tensors",
    "customize_quantize",
    "dequantize_per_tensor",
    "get_quant_params_qconv2d",
    "get_quant_params_tensor",
)

_QRANGES = {
    torch.quint8: (0, 255),
    torch.qint8: (-128, 127),
    torch.qint32: (-(2**31), 2**31 - 1),
}


def customize_quantize(x: torch.Tensor, *, scale: float, zero_point: int, dtype: torch.dtype):
    """Functional reimplementation of torch.quantize_per_tensor (per-tensor affine). Returns an integer tensor of the
    requested dtype (not a PyTorch QuantizedTensor).
    """
    if dtype not in _QRANGES:
        raise ValueError(f"Unsupported dtype {dtype}. Use torch.quint8, torch.qint8, or torch.qint32.")
    qmin, qmax = _QRANGES[dtype]

    if scale <= 0:
        raise ValueError("Scale must be a positive number.")

    if not torch.is_floating_point(x):
        raise TypeError("Input tensor must be floating point.")
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    scale_tensor = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    inv_scale = torch.reciprocal(scale_tensor)
    # PyTorch multiplies by the reciprocal of the scale before rounding to integer domain.
    q = torch.round(x * inv_scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.uint8)


def dequantize_per_tensor(q: torch.Tensor, *, scale: float, zero_point: int) -> torch.Tensor:
    """Inverse of the above: returns a float tensor."""
    return (q.to(torch.float32) - zero_point) * scale


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, log: bool = False):
    """Compare two tensors elementwise and compute overall MAE. Works safely for both float and integer tensors.

    Args:
        t1 (torch.Tensor): First tensor.
        t2 (torch.Tensor): Second tensor.
        log (bool): If True, print mismatched indices and values.

    Returns:
        float: Mean Absolute Error (MAE) between the two tensors.
    """
    if t1.shape != t2.shape:
        print(f"[Error] Shape mismatch: {t1.shape} vs {t2.shape}")
        return None

    if not torch.is_floating_point(t1):
        t1 = t1.to(torch.float32)
    if not torch.is_floating_point(t2):
        t2 = t2.to(torch.float32)

    diff = (t1 - t2).abs()

    mismatched = diff != 0

    if log:
        indices = mismatched.nonzero(as_tuple=False)
        for idx in indices:
            i = tuple(idx.tolist())
            print(f"Index {i}: t1={t1[i].item()}, t2={t2[i].item()}, diff={diff[i].item()}")

    mae = diff.mean().item()
    return mae


def compare_list(l1: list, l2: list, log: bool = False):
    """Compare two lists for each corresponding tensor and compute overall MAE.

    Args:
        l1 (torch.Tensor): First list.
        l2 (torch.Tensor): Second list.
        log (bool): If True, print mismatched indices and values.

    Returns:
        float: Average Mean Absolute Error (MAE) between the two list.
    """
    if len(l1) != len(l2):
        print(f"[Error] Length mismatch: {len(l1)} vs {len(l2)}")
        return None

    l1_clone = [t.clone() for t in l1]
    l2_clone = [t.clone() for t in l2]

    l = len(l1_clone)
    mae_list = []
    for i in range(l):
        mae_list.append(compare_tensors(l1_clone[i].dequantize(), l2_clone[i].dequantize(), log))

    if log:
        print("Mae Summary")
        for i in range(l):
            print("Mae [", i, "] =", mae_list[i])

    avg_mae = sum(mae_list) / l

    return avg_mae


QuantParams = namedtuple("QuantParams", ["scale", "zero_point", "dtype"])


def get_quant_params_tensor(tensor: torch.Tensor):
    """Retrieve quantization parameters (scale, zero_point, dtype) from a quantized tensor and return them as an object.

    Args:
        tensor (torch.Tensor): The quantized tensor.

    Returns:
        QuantParams: An object with fields .scale, .zero_point, and .dtype
    """
    if not tensor.is_quantized:
        return QuantParams(scale=None, zero_point=None, dtype=tensor.dtype)

    return QuantParams(scale=tensor.q_scale(), zero_point=tensor.q_zero_point(), dtype=tensor.dtype)


def get_quant_params_qconv2d(qconv2d: QConv2d):
    """Retrieve quantization parameters (scale, zero_point, dtype) from a quantized conv2d object and return them as an
    object.

    Args:
        qconv2d: The quantized conv2d block object.

    Returns:
        QuantParams: An object with fields .scale, .zero_point, and .dtype
    """
    if not isinstance(qconv2d, QConv2d):
        return QuantParams(scale=None, zero_point=None, dtype=None)

    return QuantParams(scale=qconv2d.scale, zero_point=qconv2d.zero_point, dtype=torch.quint8)
