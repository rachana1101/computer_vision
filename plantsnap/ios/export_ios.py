import torch
from torchvision import models
import torch.onnx

# Load YOUR trained herbalist model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 4)
model.load_state_dict(torch.load('my_herbs_model.pth'))
model.eval()

# Dummy input (224x224x3)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX (universal format)
torch.onnx.export(
    model,
    dummy_input,
    "my_herbs_model.onnx",
    input_names=['image'],
    output_names=['class'],
    opset_version=11,
    training=torch.onnx.TrainingMode.EVAL
)

print("✅ ONNX exported: my_herbs_model.onnx")
print("Next: coremltools convert my_herbs_model.onnx my_herbs.mlmodel")
