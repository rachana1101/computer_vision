import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# FULL CIFAR-10
full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
 
# TAKE ONLY FIRST 1000 IMAGES (blazing fast)
indices = np.arange(1000)
train_dataset = Subset(full_dataset, indices)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"🚀 FAST MODE: {len(train_dataset)} images, 10 classes")

# ResNet18
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 3 epochs (5 mins total)
for epoch in range(3):
    model.train()
    running_loss = 0
    for i, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 0: print(f"Batch {i}, Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1}/3 - Avg Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'plantsnap_tiny.pth')
print("💾 SAVED: plantsnap_tiny.pth")
print("🎉 DONE IN 5 MINUTES!")
