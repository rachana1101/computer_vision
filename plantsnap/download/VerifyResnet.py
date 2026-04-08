import torch
from torchvision.models import resnet18
print("✅ PyTorch imported!")
print(f"PyTorch version: {torch.__version__}")

# Create model WITHOUT pretrained (avoids SSL)
model = resnet18(weights=None)  # Modern syntax, no download
print(f"✅ ResNet18 created! {sum(p.numel() for p in model.parameters()):,} params")
print(f"Final layer: {model.fc.in_features} → 10 plants")
print("🎉 READY FOR PLANTSNAP TRAINING!")
