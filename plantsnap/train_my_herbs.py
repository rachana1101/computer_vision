import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, default_collate
import torch.nn.functional as F

# Custom collate function (fixes size issue)
def custom_collate(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack([F.interpolate(img.unsqueeze(0), size=(224,224)).squeeze(0) for img in imgs])
    labels = torch.tensor(labels)
    return imgs, labels

# Your herbs dataset
transform = transforms.Compose([
    transforms.Resize(256),  # Resize first to avoid distortion
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder('plantsnap/herbs/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)  # Small batch

print(f"✅ Found {len(train_dataset)} images across {len(train_dataset.classes)} classes")
print(f"Your classes: {train_dataset.classes}")

# ResNet18: 512 → YOUR 4 herbs
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, len(train_dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Train 10 epochs (small dataset)
print("🚀 Training YOUR herbs...")
for epoch in range(10):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/10 - Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'my_herbs_model.pth')
print("💾 SAVED: my_herbs_model.pth")
print("🎉 YOUR 4-HERB CLASSIFIER READY!")
