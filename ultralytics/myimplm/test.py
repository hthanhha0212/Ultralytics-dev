import torch

from ultralytics.myimplm import collect_results
from ultralytics.quant.utils import load_ptq_model_from_state_dict

# 1. Initialize the quantized model
base_weights = "qyolov10n.yaml"
quant_state_dict = "/home/hahuynh/Data/ProcessData/src_code/ultralytics/ultralytics/quant/quant_state_dict/qat_sttd.pt"

model = load_ptq_model_from_state_dict(base_weights=base_weights, quant_state_dict=quant_state_dict)
model.model.eval()

# 2. Generate a sample input (ensure it is divisible by 32)
input_tensor = torch.rand(1, 3, 640, 640)

# 3. Run the collector
# This performs a single inference and captures everything
results = collect_results(model, input_tensor)

# 4. Access the results
# The () at the end of the query returns the actual PyTorch tensor
l0_output = results.Layer0()  # Full output of Layer 0
l0_conv = results.Layer0.conv()  # Internal Conv output of Layer 0
l0_relu = results.Layer0.act()  # Internal ReLU output of Layer 0

# For complex layers like C2f (Layer 2)
l2_cv1 = results.Layer2.cv1()  # First internal convolution
l2_bot0 = results.Layer2.m[0]()  # First bottleneck block output
l2_bot0_c1 = results.Layer2.m[0].cv1()  # Internal conv inside the bottleneck

breakpoint()
