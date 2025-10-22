import torch
from typing import List
from torch.nn.modules.container import ModuleList 
from ultralytics.nn.modules.conv import Conv, QConcat
from ultralytics.nn.modules.block import QC2f, SCDown, SPPF, QPSA, QC2fCIB, QBottleneck, QRepVGGDW, QCIB, QAttention
from ultralytics.nn.modules.head import Qv10Detect
from ultralytics.myimplm.customqconv2d import CustomConv2d
from ultralytics.myimplm.utils_quant import  dequantize_per_tensor, get_quant_params_qconv2d 
from torch.ao.nn.quantized.modules.conv import Conv2d as QConv2d

__all__ = (
    "Generate_Custom_Conv",
    "Forward_Custom_conv2d",
    "Forward_Custom_Conv",
    "Forward_Custom_Bottleneck",
    "Forward_Custom_C2f",
    "Forward_Custom_SCDown",
    "Forward_Custom_SPPF",
    "Forward_Custom_QAttention",
    "Forward_Custom_QPSA",
    "Forward_Custom_QConcat",
    "Forward_Custom_QRepVGGDW",
    "Forward_Custom_QCIB",
    "Forward_Custom_QC2fCIB",
    "Forward_Custom_one2one_cv2",
    "Forward_Custom_one2one_cv3",
    "Forward_Custom_v10Detect",
    "Forward_Custom_Module",
)


def Generate_Custom_Conv(qconv2d_ref: QConv2d):
    """
    This function generates a customqconv2d object base on a reference quantizied conv2d object from Pytorch
    """
    custom_qconv2d = CustomConv2d(qconv2d_ref.in_channels, qconv2d_ref.out_channels, 
            qconv2d_ref.kernel_size, qconv2d_ref.stride, qconv2d_ref.padding, groups=qconv2d_ref.groups  )
    
    custom_qconv2d.sample_qparams(qconv2d_ref)

    return custom_qconv2d

def Forward_Custom_conv2d(conv2d: QConv2d, x: torch.Tensor):
    qparams = get_quant_params_qconv2d(conv2d)
    custom_conv2d = Generate_Custom_Conv(conv2d)
    res = dequantize_per_tensor(custom_conv2d(x), scale=qparams.scale, zero_point=qparams.zero_point)
    res = torch.quantize_per_tensor(res, scale=qparams.scale, zero_point=qparams.zero_point, dtype=torch.quint8)
    return res

   
def Forward_Custom_Conv(Conv_block: Conv, x: torch.Tensor):
    qparams = get_quant_params_qconv2d(Conv_block.conv)
    custom_conv2d = Generate_Custom_Conv(Conv_block.conv)
    res = dequantize_per_tensor(custom_conv2d(x), scale=qparams.scale, zero_point=qparams.zero_point)
    res = Conv_block.act(torch.quantize_per_tensor(res, scale=qparams.scale, zero_point=qparams.zero_point, dtype=torch.quint8))
    return res


def Forward_Custom_Bottleneck(Bttn_block: QBottleneck, x: torch.Tensor):
    res_cv1 = Forward_Custom_Conv(Bttn_block.cv1, x)
    res_cv2 = Forward_Custom_Conv(Bttn_block.cv2, res_cv1)
    if Bttn_block.add:
        res = Bttn_block.fl_func.add(res_cv2, x)
    else:
        res = res_cv2
    return res

def Forward_Custom_C2f(C2f_block: QC2f, x: torch.Tensor):
    res_cv1 = Forward_Custom_Conv(C2f_block.cv1, x) 
    y = list(res_cv1.chunk(2, 1))
    for m in C2f_block.m:
        res_bttn = Forward_Custom_Bottleneck(m, y[-1])
        y.append(res_bttn)
    res_cv2 = Forward_Custom_Conv(C2f_block.cv2, C2f_block.fl.cat(y, 1))
    return res_cv2

def Forward_Custom_SCDown(SCDown_block: SCDown, x: torch.Tensor):
    res_cv1 = Forward_Custom_Conv(SCDown_block.cv1, x)
    res_cv2 = Forward_Custom_Conv(SCDown_block.cv2, res_cv1)
    return res_cv2

def Forward_Custom_SPPF(SPPF_block: SPPF, x: torch.Tensor):
    res_cv1 = Forward_Custom_Conv(SPPF_block.cv1, x)
    y = [res_cv1]
    y.extend(SPPF_block.m(y[-1]) for _ in range(3))
    res_cv2 = Forward_Custom_Conv(SPPF_block.cv2, torch.cat(y, 1))
    return res_cv2

def Forward_Custom_QAttention(QAttn_block: QAttention, x:torch.Tensor):
    B, C, H, W = x.shape
    N = H * W
    res_qkv = Forward_Custom_Conv(QAttn_block.qkv, x)

    q, k, v = res_qkv.view(B, QAttn_block.num_heads, QAttn_block.key_dim * 2 + QAttn_block.head_dim, N).split(
        [QAttn_block.key_dim, QAttn_block.key_dim, QAttn_block.head_dim], dim=2
    )

    amul = QAttn_block.mul_fn.matmul(q.transpose(-2, -1), k)
    attn = QAttn_block.mul_fn.mul_scalar(amul, QAttn_block.scale)

        # If the input tensors in quint8, use the quantized softmax function, otherwise just use normal softmax
    attn = QAttn_block.sm(attn)
        
    x = QAttn_block.mul_fn.matmul(v, attn.transpose(-2, -1)).view(B, C, H, W)
    res_pe = Forward_Custom_Conv(QAttn_block.pe, v.reshape(B, C, H, W))
    
    x = QAttn_block.mul_fn.add(x, res_pe)
    res_proj = Forward_Custom_Conv(QAttn_block.proj, x)
     
    return res_proj

def Forward_Custom_QPSA(QPSA_block: QPSA, x:torch.Tensor):
    res_cv1 = Forward_Custom_Conv(QPSA_block.cv1, x)
    a, b = res_cv1.split((QPSA_block.c, QPSA_block.c), dim=1)

    res_attn = Forward_Custom_QAttention(QPSA_block.attn, b)
    b = QPSA_block.fl.add(b, res_attn)

    res_ffn0 = Forward_Custom_Conv(QPSA_block.ffn[0], b)
    res_ffn = Forward_Custom_Conv(QPSA_block.ffn[1], res_ffn0)

    b = QPSA_block.fl.add(b, res_ffn)

    res_cv2 = Forward_Custom_Conv(QPSA_block.cv2, QPSA_block.fl.cat((a, b), 1))
    return res_cv2 

def Forward_Custom_QConcat(QConcat_block: QConcat, list_ref: List[torch.Tensor]):
    """
    This block use to compare the reference library function with model computation
    """
    scale = QConcat_block.fl.scale
    zero_point = QConcat_block.fl.zero_point

    dim = QConcat_block.d
    opr = [t.clone() for t in list_ref]
    opr = [item.dequantize() for item in list_ref]
    s = torch.cat(opr, dim=dim)
    res = torch.quantize_per_tensor(s, scale=scale, zero_point=zero_point, dtype=torch.quint8)

    return res

def Forward_Custom_QRepVGGDW(RepVGGDW_block: QRepVGGDW, x: torch.Tensor):
    res_conv = Forward_Custom_conv2d(RepVGGDW_block.conv, x)
    res = RepVGGDW_block.act(res_conv)
    return res

def Forward_Custom_QCIB(QCIB_block: QCIB, x: torch.Tensor):
    res_cv1_m0 = Forward_Custom_Conv(QCIB_block.cv1[0], x)
    res_cv1_m1 = Forward_Custom_Conv(QCIB_block.cv1[1], res_cv1_m0)
    res_cv1_m2 = Forward_Custom_QRepVGGDW(QCIB_block.cv1[2], res_cv1_m1)
    res_cv1_m3 = Forward_Custom_Conv(QCIB_block.cv1[3], res_cv1_m2)
    res_cv1_m4 = Forward_Custom_Conv(QCIB_block.cv1[4], res_cv1_m3)
    if QCIB_block.add:
        res = QCIB_block.add_fn.add(res_cv1_m4, x)
    else:
        res = res_cv1_m4
    return res

def Forward_Custom_QC2fCIB(QC2fCIB_block: QC2fCIB, x: torch.Tensor):
    res_cv1 = Forward_Custom_Conv(QC2fCIB_block.cv1, x) 
    y = list(res_cv1.chunk(2, 1))
    for m in QC2fCIB_block.m:
        res_bttn = Forward_Custom_QCIB(m, y[-1])
        y.append(res_bttn)
    res_cv2 = Forward_Custom_Conv(QC2fCIB_block.cv2, QC2fCIB_block.fl.cat(y, 1))
    return res_cv2

def Forward_Custom_one2one_cv2(one2one_cv2: ModuleList, x: list, nl: int):
    res = []
    for i in range(nl):
        res0 = Forward_Custom_Conv(one2one_cv2[i][0], x[i])
        res1 = Forward_Custom_Conv(one2one_cv2[i][1], res0)
        res.append(Forward_Custom_conv2d(one2one_cv2[i][2], res1))
    return res

def Forward_Custom_one2one_cv3(one2one_cv3: ModuleList, x: list, nl: int):
    res = []
    for i in range(nl):
        res_m0_0 = Forward_Custom_Conv(one2one_cv3[i][0][0], x[i])
        res_m0_1 = Forward_Custom_Conv(one2one_cv3[i][0][1], res_m0_0)
        res_m1_0 = Forward_Custom_Conv(one2one_cv3[i][1][0], res_m0_1)
        res_m1_1 = Forward_Custom_Conv(one2one_cv3[i][1][1], res_m1_0)
        res.append(Forward_Custom_conv2d(one2one_cv3[i][2], res_m1_1))
    return res

def Forward_Custom_v10Detect(head: Qv10Detect, x: list):
    res = []
    res_one2one_cv2 = Forward_Custom_one2one_cv2(head.one2one_cv2, x, head.nl)
    res_one2one_cv3 = Forward_Custom_one2one_cv3(head.one2one_cv3, x, head.nl)
    res_one2one = [
        head.fl.cat((res_one2one_cv2[i], res_one2one_cv3[i]), 1) for i in range (head.nl)
    ]
    res_one2one = [head.dequant(item) for item in res_one2one]
    if head.training:
        return {"one2many": x, "one2one": res_one2one}
    y = head._inference(res_one2one)
    y = head.postprocess(y.permute(0, 2, 1), head.max_det, head.nc)
    return y if head.export else (y, {"one2many": x, "one2one": res_one2one})


def Forward_Custom_Module(m, x):
    """Dispatch helper that forwards inputs through the correct custom handler."""
    if isinstance(m, Conv):
        return Forward_Custom_Conv(m, x)
    if isinstance(m, QBottleneck):
        return Forward_Custom_Bottleneck(m, x)
    if isinstance(m, QC2fCIB):
        return Forward_Custom_QC2fCIB(m, x)
    if isinstance(m, QC2f):
        return Forward_Custom_C2f(m, x)
    if isinstance(m, SCDown):
        return Forward_Custom_SCDown(m, x)
    if isinstance(m, SPPF):
        return Forward_Custom_SPPF(m, x)
    if isinstance(m, QPSA):
        return Forward_Custom_QPSA(m, x)
    if isinstance(m, QRepVGGDW):
        return Forward_Custom_QRepVGGDW(m, x)
    if isinstance(m, QCIB):
        return Forward_Custom_QCIB(m, x)
    if isinstance(m, QAttention):
        return Forward_Custom_QAttention(m, x)
    if isinstance(m, Qv10Detect):
        return Forward_Custom_v10Detect(m, x)
    if isinstance(m, QConcat):
        return Forward_Custom_QConcat(m, x)
    raise TypeError(f"Unsupported module type for custom forward: {type(m)}")
