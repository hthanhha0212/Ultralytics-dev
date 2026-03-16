"""Custom Quantized 2D convolution."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.modules.utils import _pair

__all__ = ("CustomConv2d",)


class CustomConv2d(nn.Module):
    """This 2D convolution module operating on intergers inputs (uint8) and interger weights (int8) and return int32 outputs.

    The implementation performs the convolution entirely with integer arithmetic.
    Inputs are expected in NCHW format.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: Tuple[int, int] = _pair(kernel_size)
        self.stride: Tuple[int, int] = _pair(stride)
        self.padding: Tuple[int, int] = _pair(padding)
        self.dilation: Tuple[int, int] = _pair(dilation)
        self.groups = groups

        weight_shape = (
            out_channels,
            in_channels // groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        weight = torch.zeros(weight_shape, dtype=torch.int8)
        self.weight = nn.Parameter(weight, requires_grad=False)

        if bias:
            bias_param = torch.zeros(out_channels, dtype=torch.int32)
            self.bias = nn.Parameter(bias_param, requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.is_loaded = False
        # Quantization parameters
        self.scale = None
        self.zero_point = None
        self.w_scale = None
        self.w_zero_point = None
        self.x_scale = None
        self.x_zero_point = None

    def forward(self, input: Tensor) -> Tensor:
        if input.dtype == torch.quint8:
            self.x_scale = input.q_scale()
            self.x_zero_point = input.q_zero_point()
            input = input.int_repr()
        if input.dtype != torch.uint8:
            raise TypeError("input tensor must use dtype=torch.uint8.")
        if input.ndim != 4:
            raise ValueError("input tensor must have shape (N, C, H, W).")
        if input.size(1) != self.in_channels:
            raise ValueError("Input channel dimension mismatch.")
        if self.groups > 1 and self.in_channels % self.groups != 0:
            raise ValueError("Grouped convolution requires divisible in_channels.")
        
        input_int = input.to(torch.int32) - int(self.x_zero_point)
        weight_int = self.weight.to(torch.int32) - self.w_zero_point.to(torch.int32).view(-1, 1, 1, 1)

        output = self._convolution(input_int, weight_int)

        output = self.post_process_output(output)
        return output

    def _convolution(self, input_int: Tensor, weight_int: Tensor) -> Tensor:
        n, _, h_in, w_in = input_int.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        d_h, d_w = self.dilation
        p_h, p_w = self.padding

        input_padded = self._pad_input(input_int, p_h, p_w)

        out_h, out_w = self._output_dims(h_in, w_in)
        output = torch.zeros(
            (n, self.out_channels, out_h, out_w),
            dtype=torch.int32,
            device=input_int.device,
        )

        channels_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        weight_groups = weight_int.view(
            self.groups,
            out_per_group,
            channels_per_group,
            k_h,
            k_w,
        )

        for oy in range(out_h):
            h_start = oy * s_h
            h_slice = slice(
                h_start, h_start + d_h * (k_h - 1) + 1, d_h
            )
            for ox in range(out_w):
                w_start = ox * s_w
                w_slice = slice(
                    w_start, w_start + d_w * (k_w - 1) + 1, d_w
                )
                patch = input_padded[:, :, h_slice, w_slice]

                for group_idx in range(self.groups):
                    c_start = group_idx * channels_per_group
                    c_end = c_start + channels_per_group
                    o_start = group_idx * out_per_group
                    o_end = o_start + out_per_group

                    patch_group = patch[:, c_start:c_end]
                    patch_flat = patch_group.reshape(n, -1)

                    weight_group = weight_groups[group_idx].reshape(out_per_group, -1)
                    contribution = torch.matmul(
                        patch_flat, weight_group.t()
                    )
                    output[:, o_start:o_end, oy, ox] = contribution

        return output

    def _output_dims(self, h_in: int, w_in: int) -> Tuple[int, int]:
        kernel_h, kernel_w = self.kernel_size
        pad_h, pad_w = self.padding
        stride_h, stride_w = self.stride
        dil_h, dil_w = self.dilation

        out_h = (
            (h_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h
        ) + 1
        out_w = (
            (w_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w
        ) + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("Calculated output size is non-positive.")
        return out_h, out_w

    @staticmethod
    def _pad_input(input_tensor: Tensor, pad_h: int, pad_w: int) -> Tensor:
        if pad_h == 0 and pad_w == 0:
            return input_tensor
        n, c, h, w = input_tensor.shape
        padded = torch.zeros(
            (n, c, h + 2 * pad_h, w + 2 * pad_w),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        padded[:, :, pad_h : pad_h + h, pad_w : pad_w + w] = input_tensor
        return padded

    def load_int8_weight(self, weight: Tensor, bias: Optional[Tensor] = None) -> None:
        """Load integer weights (and optional bias) into the module."""
        if weight.dtype != torch.int8:
            raise TypeError("weight must use dtype=torch.int8.")
        if weight.shape != self.weight.shape:
            raise ValueError("weight shape mismatch.")
        with torch.no_grad():
            self.weight.copy_(weight)
            if bias is not None:
                if self.bias is None:
                    raise ValueError("Module was created without bias.")
                if bias.dtype != torch.int32:
                    raise TypeError("bias must use dtype=torch.int32.")
                if bias.shape != self.bias.shape:
                    raise ValueError("bias shape mismatch.")
                self.bias.copy_(bias)
        self.is_loaded = True

    def sample_qparams(self, qconv_obj: torch.ao.nn.quantized.modules.conv.Conv2d):
        """
        This method will copy all the quantization paramters from one reference object
        to itself. Additional, transfer all the weights if not loaded
        """
        self.scale = qconv_obj.scale
        self.zero_point = qconv_obj.zero_point
        self.w_scale = qconv_obj.weight().q_per_channel_scales()
        self.w_zero_point = qconv_obj.weight().q_per_channel_zero_points()
        self.bias_tensor = qconv_obj.bias().detach().to(torch.float64).view(1, -1, 1, 1)
        if not self.is_loaded:
            weight_tensor = qconv_obj.weight().int_repr()
            self.load_int8_weight(weight=weight_tensor, bias=None)

    def post_process_output(self, output: torch.Tensor):
        """
        This method process the intenger output to produce the final output 
        """
        scale_prod = (self.x_scale * self.w_scale).to(torch.float64).view(1, -1, 1, 1)
        output = output.to(torch.float64) * scale_prod + self.bias_tensor
        output = torch.round(output / self.scale) + self.zero_point
        output = output.clamp_(0, 255).to(torch.uint8)
        return output
