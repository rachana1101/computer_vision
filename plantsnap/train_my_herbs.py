import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('plantsnap/herbs/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # no custom_collate

print(f"✅ Found {len(train_dataset)} images across {len(train_dataset.classes)} classes")
print(f"Your classes: {train_dataset.classes}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False                                  # freeze backbone
model.fc = nn.Linear(512, len(train_dataset.classes))            # trainable head only
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

NUM_EPOCHS = 10
print("🚀 Training YOUR herbs...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)   # ✅ compute once, use once
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_loss)                      # ✅ called once
    new_lr = optimizer.param_groups[0]['lr']
    lr_note = f"  📉 LR → {new_lr:.6f}" if new_lr < old_lr else ""
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}{lr_note}")

torch.save(model.state_dict(), 'my_herbs_model.pth')
print("💾 SAVED: my_herbs_model.pth")
print("🎉 YOUR 4-HERB CLASSIFIER READY!")
