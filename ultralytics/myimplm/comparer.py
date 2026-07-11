from __future__ import annotations

import torch
from torch.ao.nn.quantized.modules.activation import Softmax
from torch.ao.nn.quantized.modules.functional_modules import QFunctional
from torch.nn.modules.container import ModuleList

from ultralytics.myimplm.forwarder import (
    Forward_Custom_Bottleneck,
    Forward_Custom_C2f,
    Forward_Custom_Conv,
    Forward_Custom_one2one_cv2,
    Forward_Custom_one2one_cv3,
    Forward_Custom_QAttention,
    Forward_Custom_QC2fCIB,
    Forward_Custom_QCIB,
    Forward_Custom_QConcat,
    Forward_Custom_QPSA,
    Forward_Custom_QRepVGGDW,
    Forward_Custom_SCDown,
    Forward_Custom_SPPF,
    Forward_Custom_v10Detect,
    Generate_Custom_Conv,
)
from ultralytics.myimplm.utils_quant import compare_list, compare_tensors, customize_quantize, get_quant_params_qconv2d
from ultralytics.nn.modules.block import QCIB, QPSA, SPPF, QAttention, QBottleneck, QC2f, QC2fCIB, QRepVGGDW, SCDown
from ultralytics.nn.modules.conv import Conv, QConcat
from ultralytics.nn.modules.head import Qv10Detect

__all__ = (
    "Compare_Bottleneck_Block",
    "Compare_Bottleneck_Block_v2",
    "Compare_C2f_Block",
    "Compare_Conv_Block",
    "Compare_Custom_Module",
    "Compare_Functional_block",
    "Compare_QAttention_Block",
    "Compare_QC2fCIB_Block",
    "Compare_QCIB_Block",
    "Compare_QConcat_Block",
    "Compare_QPSA_Block",
    "Compare_QRepVGGDW_Block",
    "Compare_Qv10Detect_Block",
    "Compare_SCDown_Block",
    "Compare_SPPF_Block",
    "Compare_Softmax",
    "Compare_one2one_cv2_Block",
    "Compare_one2one_cv3_Block",
    "test_conv_block",
)


def Compare_Conv_Block(conv_block: Conv, x_ref: torch.Tensor):
    """This function use to generate an equivalent custom quantized conv2d and compare the results.
    """
    res_ref = conv_block(x_ref)
    res = Forward_Custom_Conv(conv_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_Softmax(Sm: Softmax, x_ref: torch.Tensor):
    """This block use to compare the reference library function with model computation."""
    scale = Sm.scale
    zero_point = Sm.zero_point
    x = x_ref.dequantize()
    s = x.softmax(dim=-1)
    res = customize_quantize(s, scale=scale, zero_point=zero_point, dtype=torch.quint8)
    return res


def Compare_Functional_block(
    Func: QFunctional, opr1_ref: torch.Tensor, opr2_ref: torch.Tensor, list_ref: list | None = None, ops: str = ""
):
    """This block use to compare the reference library function with model computation."""
    scale = Func.scale
    zero_point = Func.zero_point
    if ops == "add":
        opr1 = opr1_ref.dequantize()
        opr2 = opr2_ref.dequantize()
        s = torch.add(opr1, opr2)
        res_ref = Func.add(opr1_ref, opr2_ref).int_repr()
        res = customize_quantize(s, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        mae = compare_tensors(res, res_ref)
    if ops == "mul":
        opr1 = opr1_ref.dequantize()
        opr2 = opr2_ref.dequantize()
        s = torch.matmul(opr1, opr2)
        res_ref = Func.matmul(opr1_ref, opr2_ref).int_repr()
        res = customize_quantize(s, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        mae = compare_tensors(res, res_ref)
    if ops == "cat":
        opr = [item.dequantize() for item in list_ref]
        s = torch.cat(opr)
        res_ref = Func.cat(list_ref).int_repr()
        res = customize_quantize(s, scale=scale, zero_point=zero_point, dtype=torch.quint8)
        mae = compare_tensors(res_ref, res)
    return mae, res_ref, res


def Compare_Bottleneck_Block(bttn_block: QBottleneck, x_ref: torch.Tensor):
    res_ref = bttn_block(x_ref)
    res = Forward_Custom_Bottleneck(bttn_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_Bottleneck_Block_v2(bttn: QBottleneck, x_ref: torch.Tensor, log: False):
    """This function use to generate an equivalent custom quantized conv2d and compare the results.
    """
    cv1_ref = bttn.cv1
    cv2_ref = bttn.cv2

    mae_cv1, res_ref_cv1, res_cv1 = Compare_Conv_Block(cv1_ref, x_ref)
    res_ref_cv2 = cv2_ref(res_ref_cv1)

    cv2_custom = Generate_Custom_Conv(cv2_ref.conv)
    qparams_cv2 = get_quant_params_qconv2d(cv2_ref.conv)
    res_cv2 = Forward_Custom_Conv(cv2_custom, res_cv1, qparams_cv2)

    mae_cv2 = compare_tensors(res_ref_cv2.dequantize(), res_cv2.dequantize())

    if bttn.add:
        res_ref = bttn.fl_func.add(x_ref, res_ref_cv2)
        res = bttn.fl_func.add(x_ref, res_cv2)
        mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    else:
        res_ref = res_ref_cv2
        res = res_cv2
        mae = mae_cv2

    if log:
        print("mae_cv1", mae_cv1)
        print("mae_cv2", mae_cv2)

    return mae, res_ref, res


def Compare_C2f_Block(c2f_block: QC2f, x_ref: torch.Tensor):
    res_ref = c2f_block(x_ref)
    res = Forward_Custom_C2f(c2f_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_SCDown_Block(SCDown_block: SCDown, x_ref: torch.Tensor):
    res_ref = SCDown_block(x_ref)
    res = Forward_Custom_SCDown(SCDown_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_SPPF_Block(SPPF_block: SPPF, x_ref: torch.Tensor):
    res_ref = SPPF_block(x_ref)
    res = Forward_Custom_SPPF(SPPF_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_QConcat_Block(QConcat_block: QConcat, x_ref: list[torch.Tensor]):
    res_ref = QConcat_block(x_ref)
    res = Forward_Custom_QConcat(QConcat_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_QRepVGGDW_Block(RepVGGDW_block: QRepVGGDW, x_ref: torch.Tensor):
    res_ref = RepVGGDW_block(x_ref)
    res = Forward_Custom_QRepVGGDW(RepVGGDW_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_QCIB_Block(QCIB_block: QCIB, x_ref: torch.Tensor):
    res_ref = QCIB_block(x_ref)
    res = Forward_Custom_QCIB(QCIB_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_QC2fCIB_Block(QC2fCIB_block: QC2fCIB, x_ref: torch.Tensor):
    res_ref = QC2fCIB_block(x_ref)
    res = Forward_Custom_QC2fCIB(QC2fCIB_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_QPSA_Block(QPSA_block: QPSA, x_ref: torch.Tensor):
    res_ref = QPSA_block(x_ref)
    res = Forward_Custom_QPSA(QPSA_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_QAttention_Block(QAttn_block: QAttention, x_ref: torch.Tensor):
    res_ref = QAttn_block(x_ref)
    res = Forward_Custom_QAttention(QAttn_block, x_ref)
    mae = compare_tensors(res_ref.dequantize(), res.dequantize())
    return mae, res_ref, res


def Compare_one2one_cv2_Block(one2one_cv2_block: ModuleList, x_ref: list, nl: int):
    res_ref = [one2one_cv2_block[i](x_ref[i]) for i in range(nl)]
    res = Forward_Custom_one2one_cv2(one2one_cv2_block, x_ref, nl)
    avg_mae = compare_list(res_ref, res)
    return avg_mae, res_ref, res


def Compare_one2one_cv3_Block(one2one_cv3_block: ModuleList, x_ref: list, nl: int):
    res_ref = [one2one_cv3_block[i](x_ref[i]) for i in range(nl)]
    res = Forward_Custom_one2one_cv3(one2one_cv3_block, x_ref, nl)
    avg_mae = compare_list(res_ref, res)
    return avg_mae, res_ref, res


def Compare_Qv10Detect_Block(head: Qv10Detect, x_ref: list):
    x = [t.clone() for t in x_ref]
    res_ref = head(x_ref)
    res = Forward_Custom_v10Detect(head, x)

    if head.training:
        res_one2one_ref = res_ref["one2one"]
        res_one2one = res["one2one"]
        avg_mae = compare_list(res_one2one_ref, res_one2one)
        return avg_mae, res_one2one_ref, res_one2one

    else:
        res_one2one_ref = res_ref[1]["one2one"]
        res_one2one = res[1]["one2one"]
        avg_y = compare_tensors(res_ref[0].dequantize(), res[0].dequantize())
        avg_mae = compare_list(res_one2one_ref, res_one2one)
        return avg_mae, avg_y, res_one2one_ref, res_one2one


def Compare_Custom_Module(m, x_ref):
    """Dispatch helper that compares a module output with its custom counterpart."""
    if isinstance(m, Conv):
        return Compare_Conv_Block(m, x_ref)
    if isinstance(m, QBottleneck):
        return Compare_Bottleneck_Block(m, x_ref)
    if isinstance(m, QC2fCIB):
        return Compare_QC2fCIB_Block(m, x_ref)
    if isinstance(m, QC2f):
        return Compare_C2f_Block(m, x_ref)
    if isinstance(m, SCDown):
        return Compare_SCDown_Block(m, x_ref)
    if isinstance(m, SPPF):
        return Compare_SPPF_Block(m, x_ref)
    if isinstance(m, QPSA):
        return Compare_QPSA_Block(m, x_ref)
    if isinstance(m, QRepVGGDW):
        return Compare_QRepVGGDW_Block(m, x_ref)
    if isinstance(m, QCIB):
        return Compare_QCIB_Block(m, x_ref)
    if isinstance(m, QAttention):
        return Compare_QAttention_Block(m, x_ref)
    if isinstance(m, Qv10Detect):
        return Compare_Qv10Detect_Block(m, x_ref)
    if isinstance(m, QConcat):
        return Compare_QConcat_Block(m, x_ref)
    raise TypeError(f"Unsupported module type for custom comparison: {type(m)}")


def test_conv_block(conv_block, model, num_test=10):
    """Test the conv_block multiple times using random inputs.

    Args:
        conv_block: The convolution block to test.
        model: The model that provides a .quant() method for quantization.
        num_test (int): Number of tests to run.

    Returns:
        dict: A summary with total, passed, failed, and pass rate.
    """
    passed = 0
    failed = 0

    for i in range(num_test):
        x = torch.rand(1, 3, 224, 224)
        x_quant = model.quant(x)

        try:
            result = Compare_Conv_Block(conv_block, x_quant)
            if result:
                passed += 1
                print(f"[{i + 1}/{num_test}] ✅ Passed")
            else:
                failed += 1
                print(f"[{i + 1}/{num_test}] ❌ Failed")
        except Exception as e:
            failed += 1
            print(f"[{i + 1}/{num_test}] ⚠️ Error: {e}")

    summary = {"total": num_test, "passed": passed, "failed": failed, "pass_rate": passed / num_test * 100}

    print("\n===== Test Summary =====")
    print(f"Total: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass Rate: {summary['pass_rate']:.2f}%")

    return summary
