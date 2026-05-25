import torch
from ultralytics import YOLO

print("Initializing the NPU-friendly YOLOv5 model...")
model = YOLO('ultralytics/cfg/models/v5/npu_friendly_yolov5n.yaml')

print("Creating dummy tensor of shape (1, 3, 640, 640)...")
dummy_input = torch.randn(1, 3, 640, 640)

print("Running forward pass...")
try:
    # Forward pass
    # model.predict returns results; model.model(x) runs the raw torch graph
    outputs = model.model(dummy_input)
    
    if isinstance(outputs, tuple) or isinstance(outputs, list):
        print(f"Forward pass successful! Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            if isinstance(out, torch.Tensor):
                print(f" Output {i} shape: {out.shape}")
            else:
                print(f" Output {i} type: {type(out)}")
    else:
        print(f"Forward pass successful! Output shape: {outputs.shape}")

except Exception as e:
    print(f"Forward pass failed with exception: {e}")
    import traceback
    traceback.print_exc()
