import coremltools as ct
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

CLASS_LABELS = [
    'acanthaceae', 'ashwagandha', 'asian ginseng', 'astragalus',
    'basil', 'birch', 'black cohosh', 'black haw', 'black pepper',
    'black walnut', 'burdock', 'calendula', 'california poppy',
    'catnip herb', 'chamomile', 'chaste tree', 'chickweed', 'comfrey',
    'coriander', 'cramp bark', 'cumin', 'dandelion', 'echinacea',
    'elder berry', 'elder berry flower', 'elecampane', 'eleuthero',
    'fennel', 'feverfew', 'garlic', 'ginger', 'ginger root',
    'ginko leaf', 'green tea', 'holy basil', 'hops', 'lady\'s mantle',
    'lavender', 'lemon balm', 'licorice root', 'linden', 'meadowsweet',
    'motherwort', 'mullein', 'nettle', 'nutmeg', 'oak', 'orange',
    'oregano', 'passionflower', 'peppermint', 'plantain leaf',
    'raspberry', 'red clover', 'reishi', 'rosemary', 'sage',
    'saw palmetto', 'shepherd\'s purse', 'shiitake', 'skullcap',
    'spilanthes', 'st. john\'s wort', 'thyme', 'tulsi', 'turmeric',
    'valerian', 'vervain', 'white pine', 'wild yam'  # ← double-check 'yarrow' is in your dataset
]

print(f"Total classes: {len(CLASS_LABELS)}")  # should print 70

# ---- Build & load model ----
class NormalizedResNet(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.model = resnet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x =  self.model(x)
        return F.softmax(x, dim=1)

backbone = models.resnet18(weights=None)
backbone.fc = nn.Linear(512, 70)  # ✅ 70 classes
backbone.load_state_dict(torch.load('my_herbs_model.pth', map_location='cpu'))

model = NormalizedResNet(backbone)
model.eval()

# ---- Trace ----
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# ---- Convert ----
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=example_input.shape, scale=1/255.0)],
    classifier_config=ct.ClassifierConfig(CLASS_LABELS)
)

mlmodel.save('my_herbs.mlpackage')
print("✅ iOS ready: my_herbs.mlpackage (Drag to Xcode!)")
