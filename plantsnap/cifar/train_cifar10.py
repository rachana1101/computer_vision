import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# CIFAR-10: 50k train, 10k test, 10 classes (perfect!)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"✅ CIFAR-10 loaded: {len(train_dataset)} images, 10 classes")
print("Classes:", train_dataset.classes)

# ResNet18: 512 → 10 classes
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Train 3 epochs
print("🚀 Training starts...")
for epoch in range(3):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), 'plantsnap_model.pth')
print("💾 Model saved: plantsnap_model.pth")
print("🎉 TRAINING COMPLETE!")
