import torch
from ultralytics import YOLO
from ultralytics.quant import load_ptq_model_from_state_dict


# This initialzied the quantized model, this model get a total of 23 layers (might not be correct)
model = load_ptq_model_from_state_dict(
    base_weights = 'qyolov10n.yaml',
    quant_state_dict = '/home/hahuynh/Data/ProcessData/src_code/ultralytics/ultralytics/quant/quant_state_dict/qat_sttd.pt'
)

# Use the variable model to print the model
model

# Get the first layer of the model 
Conv_1 =  model.model.model[1]

# This Conv will contain a conv2d block and a RELU
q_conv2d = Conv_1.conv
act = Conv_1.act

# Suppose there is a image or tensor named input
input = torch.rand(1, 3, 224, 224)

# Normally this model is able to process end-to-end with this input
output = model(input)

# However I want to collect the result for each stage of the model, let's call this infer_result
# and this infer_result is generated from the intended develop function like 
# infer_result = collector(model, input)
# So if I want to get the result of the first layer it might be
# res = infer_result.Layer0
# And the res is actually is res = model.model.model[0](input)
# And I want the result should be more details, like
# infer_result.Layer0.conv (return the quantized conv2d result from Conv block)

# Also noted, for the input to be able processed, it should be quanized before feeding to the block 
input_quant = model.quant(input)

# Now it is able to feed to the Conv block
Infer_result.Layer0.conv = model.model.model[0](input_quant)

# In the case if the input is already a quantized format, no need to use model.quant()


breakpoint()