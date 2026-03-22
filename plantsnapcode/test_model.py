import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# CIFAR-10 classes (your trained model knows these)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load your trained model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 10)
model.load_state_dict(torch.load('plantsnap_tiny.pth'))
model.eval()

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Test with sample cat image (download)
url = 'https://github.com/pytorch/hub/raw/master/images/dog.jpg'
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Preprocess + predict
img_t = transform(img).unsqueeze(0)
with torch.no_grad():
    outputs = model(img_t)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top_prob, top_catid = torch.topk(probs, 1)

print(f"Prediction: {classes[top_catid]} (conf: {top_prob.item():.2%})")
print("Top 3:", [(classes[i], probs[i].item()) for i in torch.topk(probs, 3).indices])

plt.imshow(img)
plt.title(f"Predicted: {classes[top_catid]} ({top_prob.item():.2%})")
plt.show()
