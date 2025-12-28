from __future__ import annotations

from typing import Tuple

import torch
from torch.nn.modules.utils import _pair

__all__ = ("has_insufficient_window",)


def _get_padding(padding) -> Tuple[int, int]:
    if isinstance(padding, str):
        raise ValueError("String padding modes are not supported for this check.")
    return _pair(padding)


def has_insufficient_window(conv2d: torch.nn.Module, x: torch.Tensor) -> bool:
    """
    Return True if the convolution would drop a trailing partial window.

    This happens when the effective input size (including padding) does not align
    with the stride, leaving a remainder smaller than the kernel.
    """
    if x.ndim != 4:
        raise ValueError("Input tensor must have shape (N, C, H, W).")

    k_h, k_w = _pair(conv2d.kernel_size)
    s_h, s_w = _pair(conv2d.stride)
    p_h, p_w = _get_padding(conv2d.padding)
    d_h, d_w = _pair(conv2d.dilation)

    h_in, w_in = x.shape[-2:]
    k_eff_h = d_h * (k_h - 1) + 1
    k_eff_w = d_w * (k_w - 1) + 1

    avail_h = h_in + 2 * p_h - k_eff_h
    avail_w = w_in + 2 * p_w - k_eff_w

    if avail_h < 0 or avail_w < 0:
        return True

    return (avail_h % s_h != 0) or (avail_w % s_w != 0)
