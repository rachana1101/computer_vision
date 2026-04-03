import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# YOUR 4 herbs (alphabetical order)
classes = ['gotukola', 'marigold', 'mullein', 'yarrow']

# Load YOUR trained model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 4)  # YOUR 4 classes
model.load_state_dict(torch.load('my_herbs_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Test any image from your herbs folder
img_path = 'plantsnap/herbs/val/yarrow/c10005_2.jpg'  # CHANGE TO YOUR IMAGE
img = Image.open(img_path)

img_t = transform(img).unsqueeze(0)
with torch.no_grad():
    outputs = model(img_t)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_class = torch.topk(probs, 1)

print(f"✅ PREDICTION: {classes[top_class]} ({top_prob.item():.1%})")
print("Top 3:", [(classes[i], probs[i].item()) for i in torch.topk(probs, 3).indices])

plt.imshow(img)
plt.title(f"Predicted: {classes[top_class]} ({top_prob.item():.1%})")
plt.show()
