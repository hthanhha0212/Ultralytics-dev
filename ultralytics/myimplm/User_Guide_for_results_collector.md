# User Guide: Intermediate Results Collector

This feature provides a simple and powerful way to collect all intermediate results from a quantized model during a single inference run. If your model has 24 layers (like YOLOv10n), it will capture the output of every single layer simultaneously.

Additionally, it captures sub-intermediate results from within each block. For example, a `Conv` block consists of a `quantized2d conv` and a `ReLU` activation; this tool allows you to capture the results of both components individually.

### 1. Feature Summary
The `ResultsCollector` hooks into the model architecture and records the output tensor of every module and sub-module.
- **Complete Coverage**: Captures all layers and their internal components.
- **Easy Access**: Uses intuitive dot notation (e.g., `results.Layer0.conv()`).
- **Quantization Aware**: Correctly handles quantized tensors and in-place operations like ReLU.
- **Zero Configuration**: Automatically maps the model structure without manual setup.

### 2. Usage Example
This example shows how to import the collector, initialize your quantized model, and query specific intermediate results.

```python
import torch
from ultralytics.quant.utils import load_ptq_model_from_state_dict
from ultralytics.myimplm import collect_results

# 1. Initialize the quantized model
base_weights = 'qyolov10n.yaml'
quant_state_dict = 'path/to/your/qat_sttd.pt'

model = load_ptq_model_from_state_dict(
    base_weights = base_weights,
    quant_state_dict = quant_state_dict
)
model.model.eval()

# 2. Generate a sample input (ensure it is divisible by 32)
input_tensor = torch.rand(1, 3, 640, 640)

# 3. Run the collector
# This performs a single inference and captures everything
results = collect_results(model, input_tensor)

# 4. Access the results
# The () at the end of the query returns the actual PyTorch tensor
l0_output = results.Layer0()           # Full output of Layer 0
l0_conv   = results.Layer0.conv()      # Internal Conv output of Layer 0
l0_relu   = results.Layer0.act()       # Internal ReLU output of Layer 0

# For complex layers like C2f (Layer 2)
l2_cv1    = results.Layer2.cv1()       # First internal convolution
l2_bot0   = results.Layer2.m[0]()      # First bottleneck block output
l2_bot0_c1 = results.Layer2.m[0].cv1() # Internal conv inside the bottleneck

print(f"Layer 0 Conv Shape: {l0_conv.shape}")
```

### 3. Full Query Table (YOLOv10n Full Architecture)
This table provides the exact commands needed to query every layer and all available sub-components in the YOLOv10n model.

| Layer | Component Path (Recursive) | Query Command (Tensor Access) |
| :--- | :--- | :--- |
| **0** | **Conv** | `results.Layer0()` |
| | └── conv (Conv2d) | `results.Layer0.conv()` |
| | └── act (ReLU) | `results.Layer0.act()` |
| **1** | **Conv** | `results.Layer1()` |
| | └── conv (Conv2d) | `results.Layer1.conv()` |
| | └── act (ReLU) | `results.Layer1.act()` |
| **2** | **QC2f** | `results.Layer2()` |
| | ├── cv1 (Conv) | `results.Layer2.cv1()` |
| | ├── cv2 (Conv) | `results.Layer2.cv2()` |
| | └── m[0] (QBottleneck) | `results.Layer2.m[0]()` |
| | &nbsp;&nbsp;&nbsp;&nbsp;├── cv1 (Conv) | `results.Layer2.m[0].cv1()` |
| | &nbsp;&nbsp;&nbsp;&nbsp;└── cv2 (Conv) | `results.Layer2.m[0].cv2()` |
| **3** | **Conv** | `results.Layer3()` |
| | └── conv (Conv2d) | `results.Layer3.conv()` |
| | └── act (ReLU) | `results.Layer3.act()` |
| **4** | **QC2f** | `results.Layer4()` |
| | ├── cv1, cv2 | `results.Layer4.cv1()`, `results.Layer4.cv2()` |
| | └── m[0], m[1] | `results.Layer4.m[0]()`, `results.Layer4.m[1]()` |
| **5** | **SCDown** | `results.Layer5()` |
| | ├── cv1 (Conv) | `results.Layer5.cv1()` |
| | └── cv2 (Conv) | `results.Layer5.cv2()` |
| **6** | **QC2f** | `results.Layer6()` |
| | ├── cv1, cv2 | `results.Layer6.cv1()`, `results.Layer6.cv2()` |
| | └── m[0], m[1] | `results.Layer6.m[0]()`, `results.Layer6.m[1]()` |
| **7** | **SCDown** | `results.Layer7()` |
| | ├── cv1, cv2 | `results.Layer7.cv1()`, `results.Layer7.cv2()` |
| **8** | **QC2f** | `results.Layer8()` |
| | ├── cv1, cv2 | `results.Layer8.cv1()`, `results.Layer8.cv2()` |
| | └── m[0] | `results.Layer8.m[0]()` |
| **9** | **SPPF** | `results.Layer9()` |
| | ├── cv1, cv2 | `results.Layer9.cv1()`, `results.Layer9.cv2()` |
| | └── m (MaxPool2d) | `results.Layer9.m()` |
| **10** | **QPSA** | `results.Layer10()` |
| | ├── cv1, cv2 | `results.Layer10.cv1()`, `results.Layer10.cv2()` |
| | ├── attn (QAttention) | `results.Layer10.attn()` |
| | │&nbsp;&nbsp;&nbsp;├── qkv, proj, pe | `results.Layer10.attn.qkv()`, ... |
| | │&nbsp;&nbsp;&nbsp;└── sm (Softmax) | `results.Layer10.attn.sm()` |
| | └── ffn (Sequential) | `results.Layer10.ffn()` |
| | &nbsp;&nbsp;&nbsp;&nbsp;└── 0, 1 (Conv) | `results.Layer10.ffn[0]()`, `results.Layer10.ffn[1]()` |
| **11** | **Upsample** | `results.Layer11()` |
| **12** | **QConcat** | `results.Layer12()` |
| **13** | **QC2f** | `results.Layer13()` |
| | ├── cv1, cv2, m[0] | `results.Layer13.cv1()`, `results.Layer13.cv2()`, ... |
| **14** | **Upsample** | `results.Layer14()` |
| **15** | **QConcat** | `results.Layer15()` |
| **16** | **QC2f** | `results.Layer16()` |
| **17** | **Conv** | `results.Layer17()` |
| **18** | **QConcat** | `results.Layer18()` |
| **19** | **QC2f** | `results.Layer19()` |
| **20** | **SCDown** | `results.Layer20()` |
| **21** | **QConcat** | `results.Layer21()` |
| **22** | **QC2fCIB** | `results.Layer22()` |
| | ├── cv1, cv2 | `results.Layer22.cv1()`, `results.Layer22.cv2()` |
| | └── m[0] (QCIB) | `results.Layer22.m[0]()` |
| | &nbsp;&nbsp;&nbsp;&nbsp;└── cv1 (Sequential) | `results.Layer22.m[0].cv1()` |
| | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── 0, 1, 3, 4 (Conv) | `results.Layer22.m[0].cv1[0]()`, ... |
| **23** | **Qv10Detect** | `results.Layer23()` |
| | ├── one2one_cv2[0..2] | `results.Layer23.one2one_cv2[0]()`, ... |
| | ├── one2one_cv3[0..2] | `results.Layer23.one2one_cv3[0]()`, ... |
| | └── dfl (DFL) | `results.Layer23.dfl()` |

**Note**: To explore any layer in more detail, simply print the results object: `print(results.Layer22)`. This will display the full nested structure of that specific layer.
