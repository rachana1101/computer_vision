import coremltools as ct
import torch
from torchvision import models

# Load your PyTorch model directly (skip ONNX)
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 4)
model.load_state_dict(torch.load('my_herbs_model.pth'))
model.eval()

# Trace model with dummy input
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Convert PyTorch → CoreML (modern API)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=example_input.shape, scale=1/255.0)],
    classifier_config=ct.ClassifierConfig(class_labels=['gotukola', 'marigold', 'mullein', 'yarrow'])
)

mlmodel.save('my_herbs.mlpackage')
print("✅ iOS ready: my_herbs.mlmodel (Drag to Xcode!)")
