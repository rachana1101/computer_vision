# ***

# ## What You Did: PlantSnap Training Summary

# ### The Project
# You trained a **ResNet18 model** on **3,461 herb images across 70 classes** using PyTorch, then iteratively improved training by tuning hyperparameters.

# ***

# ### Problem 1: NumPy Compatibility Error
# **What happened:** Training crashed with `RuntimeError: Numpy is not available`.

# **Root cause:** NumPy 2.0 changed its C API, breaking binary compatibility with PyTorch which was compiled against NumPy 1.x.

# **Fix:**
# ```bash
# pip install "numpy<2"
# ```

# **Interview answer:**
# > "I hit a dependency conflict between NumPy 2.0 and PyTorch. NumPy 2.0 broke the binary API PyTorch relied on. I downgraded to NumPy 1.26.4 which resolved it."

# ***

# ### Problem 2: Loss Dropping Too Slowly

# **What you observed across 24 epochs:**

# | Epoch | Loss | Key Observation |
# |-------|------|----------------|
# | 1 | 4.6567 | Near random baseline (ln 70 = 4.25) |
# | 10 | 4.2033 | Just crossed random baseline |
# | 24 | 3.9834 | Steady but very slow |

# **Root cause:** Learning rate was too low → model taking tiny steps each epoch (~0.013 drop/epoch).

# **Fix 1 — Increase epochs to 50:**
# ```python
# num_epochs = 50
# ```

# **Fix 2 — Add ReduceLROnPlateau scheduler:**
# ```python
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=3, verbose=True
# )
# scheduler.step(epoch_loss)  # at end of each epoch
# ```

# **Why this helps:**
# - The scheduler **automatically halves the learning rate** when loss stops improving for 3 consecutive epochs.
# - This allows the model to take big steps early (fast learning) and small steps later (fine-tuning), instead of tiny steps throughout.

# ***

# ### Interview Explanation (Deliver This)

# > "I trained ResNet18 on 3,461 herb images across 70 classes. Initially, loss was near the random baseline of ln(70)=4.25, which means the model was essentially guessing. After 24 epochs, loss dropped to 3.98, but the pace was too slow — about 0.013 per epoch. At that rate I'd need 100+ epochs to reach meaningful accuracy.
# >
# > The fix had two parts: I increased epochs to 50 and added a ReduceLROnPlateau scheduler. The scheduler monitored loss and halved the learning rate whenever improvement stalled for 3 consecutive epochs. This is important because a fixed learning rate is too aggressive for fine-tuning but too slow early on. The adaptive scheduler handles both phases automatically.
# >
# > I also hit a NumPy 2.0 compatibility issue with PyTorch mid-training, which I resolved by pinning NumPy to 1.26.4."

# ***

# ### Key Concepts to Know for Follow-up Questions

# **"Why not just use a very high learning rate from the start?"**
# > "High LR causes loss to oscillate or diverge. We saw minor upticks at epochs 10–11 and 17–18 even at moderate LR. The scheduler gives you the best of both worlds."

# **"What is an epoch?"**
# > "One complete pass through all 3,461 training images. Each epoch the model sees every herb once, updates weights once per batch, and ideally gets a bit better."

# **"Why ln(70) as the random baseline?"**
# > "With 70 equally likely classes, a random model assigns 1/70 probability to each. Cross-entropy of uniform random guessing = −log(1/70) = ln(70) ≈ 4.25."

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ============================================================
# STEP 1: IMAGE PREPROCESSING PIPELINE
# Every herb image goes through these steps before training
# ============================================================
# Why Resize(256) BEFORE CenterCrop(224)?
# Images come in all different sizes from the field
# Resize first standardizes ALL images to same scale
# THEN crop consistently — otherwise small images
# would lose too much content with direct crop

# Why 224x224 specifically?

# ResNet18 was designed and trained on ImageNet at 224x224
# Feeding different size = wrong results
# It's ResNet18's "native resolution"

transform = transforms.Compose([
    # Step 1: Resize shortest side to 256px
    # reduces computation, keeps aspect ratio
    transforms.Resize(256),
    
    # Step 2: Crop center 224x224
    # removes edge noise, keeps main subject
    # 224x224 = ResNet18's expected input size
    transforms.CenterCrop(224),
    
    # Step 3: Convert PIL image to PyTorch tensor
    # scales pixel values from [0,255] to [0,1]
    transforms.ToTensor(),
    
    # Step 4: Normalize using ImageNet mean & std
    # centers data around zero for ResNet18 comfort zone
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])


# ============================================================
# STEP 2: DATA LOADING WITH MINI-BATCHES
# ============================================================
train_dataset = datasets.ImageFolder('plantsnap/herbs/train', transform=transform)
# shuffle=True — Your instinct is correct:
# Without shuffle:
# Epoch 1: basil, basil, basil... chamomile, chamomile...
# Epoch 2: basil, basil, basil... chamomile, chamomile...
# Model learns herbs in order → biased learning → thinks basil always comes first
# With shuffle:
# Epoch 1: chamomile, basil, nettle, lavender, basil...
# Epoch 2: nettle, chamomile, basil, lavender, nettle...
# Random order every epoch → unbiased learning → generalizes better ✅

# batch_size=16 — Why 16 specifically?
# Remember our mini-batch discussion?
# Too large (512+) → needs more GPU memory 😰
# Too small (1-2)  → too noisy, unstable training 😬
# Sweet spot       → 16, 32, 64, 128 ✅
# Why 16 specifically for the project?

# I trained on Mac with MPS backend
# 16 is conservative for laptop GPU memory
# Safe choice for variable herb image dataset
# Could try 32 if you have memory available

train_loader = DataLoader(
    train_dataset, 
    # Mini-batch size — feeds 16 images at a time
    # small enough for Mac MPS memory
    # large enough for stable gradient updates
    batch_size=16, 
    # Shuffle every epoch to prevent ordered bias
    # ensures model sees herbs in random order
    # helps generalization to new herb images
    shuffle=True
)



#Epoch = One complete pass through your ENTIRE dataset
#50 images × 70 classes = 3,500 total images
# 3,500 images ÷ batch_size 16 = 219 batches per epoch
# 219 batches × 10 epochs = 2,190 total weight updates
# Less than 50    → too little, model struggles 😬
# 50 images       → minimal but workable ✅
# 100-200 images  → comfortable zone 👍
# 1000+ images    → ideal 🎯

# 50 per class works BECAUSE:

# You used transfer learning ✅
# ResNet18 already knew edges, textures, shapes from ImageNet
# You only taught it the difference between herbs
# Frozen backbone needed very little data to fine-tune ✅

print(f"✅ Found {len(train_dataset)} images across {len(train_dataset.classes)} classes")
print(f"Your classes: {train_dataset.classes}")

# ============================================================
# STEP 3: DEVICE SETUP — USE MAC GPU (MPS)
# ============================================================

# MPS = Metal Performance Shaders — Apple Silicon GPU acceleration
# Falls back to CPU if MPS not available
# Significantly faster than CPU for matrix operations in neural nets
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================
# STEP 4: MODEL SETUP — TRANSFER LEARNING
# ============================================================
# Load ResNet18 pretrained on 1.2M ImageNet images
# Already knows edges, textures, shapes, leaf patterns from ImageNet
model = models.resnet18(weights='IMAGENET1K_V1')


# ❄️ FREEZE all 18 layers (the "backbone")
# requires_grad=False = backpropagation won't update these weights
# Preserves all ImageNet knowledge — don't throw away 1.2M image learning!
# Critical for small dataset (50 images/class) — training all layers = overfitting disaster
for param in model.parameters():
    param.requires_grad = False     
    

# 🔥 Replace ONLY the final classification layer (fc = fully connected)
# 512 = ResNet18's feature output size (fixed architecture)
# len(train_dataset.classes) = 70 herb classes
# This is the ONLY layer that learns to distinguish your herbs
# Replaces original ImageNet output (1000 classes) with our herbs (70 classes)     
# 
# 512 is the number of features ResNet18 outputs from its backbone.

# Think of it this way — after your herb image travels through all 17 frozen layers, ResNet18 summarizes everything it saw into 512 numbers:
# Your herb image (224x224x3 pixels)
#          ↓
# Layer 1:  detects edges
#          ↓
# Layer 2:  detects textures
#          ↓
#    ... 15 more layers ...
#          ↓
# Layer 17: outputs 512 numbers  ← this is the backbone output
#          ↓
# Layer 18: fc = nn.Linear(512, 70)  ← your final layer
#          ↓
# 70 confidence scores (one per herb class)

# Those 512 numbers represent things like:

# How much "leaf-ness" is in the image
# How much "green texture" was detected
# How much "circular shape" was found
# etc. (512 such features)


# Why 512 specifically?
# That's just how ResNet18 was designed by its creators. It's the architecture's fixed output size:
# ResNet18  → 512 features
# ResNet50  → 2048 features  (bigger model, more features)
# ResNet152 → 2048 features  (even bigger)
# So your final layer is essentially saying:
# "Take these 512 feature measurements and map them to 70 herb probabilities"

# In interview language:
# "512 is the output dimension of ResNet18's final average pooling layer — the backbone's feature vector size. My classification head takes that 512-dimensional feature vector and maps it to 70 class probabilities using a single linear transformation."
#                        
model.fc = nn.Linear(512, len(train_dataset.classes))
model = model.to(device)


# ============================================================
# STEP 5: LOSS, OPTIMIZER, SCHEDULER
# ============================================================
# Loss function for multi-class classification (70 herb classes)
# Measures how wrong the prediction is
# Output: single number — higher = more wrong, lower = more right
# "How wrong was my prediction, and by how much?"

# "I used CrossEntropyLoss because PlantSnap is a 70-class classification problem.
# It measures how far the predicted probability distribution is from the true label, 
# giving larger penalties for confident wrong predictions and smaller penalties when the model is close. 
# It works together with the softmax in my NormalizedResNet to produce proper probability scores across all 70 herb classes."
criterion = nn.CrossEntropyLoss()

# Adam optimizer — smarter than basic gradient descent
# model.fc.parameters() = ONLY update the final layer (not frozen backbone)
# lr=0.001 = starting learning rate (step size going downhill)
# Adam adapts learning rate automatically per parameter


# torch.optim.Adam
# Adam = Adaptive Moment estimation
# Fancy name for a smart version of gradient descent that:

# Automatically adjusts the learning rate for each weight individually
# Weights that haven't changed much → get a bigger nudge
# Weights that change a lot → get a smaller nudge
# Much smarter than basic SGD which treats all weights equally

# Basic SGD:  same learning rate for ALL weights  😐
# Adam:       custom learning rate per weight      😎

# model.fc.parameters()
# Remember frozen backbone? This is where it matters!
# model.parameters()     → ALL 18 layers  ❌ don't want this
# model.fc.parameters()  → ONLY final layer ✅ just your herb classifier
# You are telling Adam:
# "Only update the weights in the final layer — leave the frozen backbone alone"
# If you had written model.parameters() here, Adam would try to update frozen weights — wasted computation.

# lr=0.001
# The starting learning rate — size of each step downhill.
# 0.001 = safe default for Adam with transfer learning
# Too high (0.1)   → overshoots, training goes crazy
# Too low (0.00001)→ trains but extremely slowly
# 0.001            → well established sweet spot ✅
# And remember your scheduler then automatically reduces this when loss plateaus!
# Start:       lr = 0.001
# After plateau: lr = 0.0005  (halved)
# After plateau: lr = 0.00025 (halved again)

# The full picture of how these work together:
# CrossEntropyLoss    → measures HOW WRONG the prediction was
# Backpropagation     → calculates WHICH weights caused the error
# Adam optimizer      → decides HOW MUCH to adjust each weight
# lr=0.001            → controls the SIZE of each adjustment
# ReduceLROnPlateau   → shrinks lr when progress stalls
# They are all one connected system! 🎯

# Interview answer:
# "I used Adam optimizer because it adapts the learning rate individually for each weight, 
# which converges faster than vanilla SGD especially with transfer learning.
# I passed model.fc.parameters() specifically rather than all parameters because the backbone is frozen
# — no point optimizing weights that aren't being updated. Starting lr of 0.001 is the 
# standard sweet spot for Adam with pretrained models, and my ReduceLROnPlateau scheduler handles decay automatically."

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Automatically reduces learning rate when training plateaus
# mode='min' = watching for loss to stop decreasing
# factor=0.5 = cut learning rate in half when plateau detected
# patience=3 = wait 3 epochs before deciding it's a plateau
# This is automated solution for "model stopped improving" problem
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)


# ============================================================
# STEP 6: TRAINING LOOP
# ============================================================
NUM_EPOCHS = 10

# Epoch = one complete pass through ALL 3,500 herb images
# 10 epochs = model sees every herb image 10 times
# Each epoch: different random order (shuffle=True)
# When loss stops decreasing → model has converged

print("🚀 Training YOUR herbs...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode (enables dropout etc.)
    running_loss = 0

    for imgs, labels in train_loader:   # Each iteration = 1 mini-batch of 16 images
        imgs, labels = imgs.to(device), labels.to(device)

        # Clear gradients from previous batch
        # Without this: gradients accumulate and training goes wrong        

        # PyTorch has a quirky design decision — gradients ACCUMULATE by default.
        # python# Without zero_grad():
        # Batch 1: gradient for weight = 0.3
        # Batch 2: gradient for weight = 0.3 + 0.5 = 0.8  ← accumulated!
        # Batch 3: gradient for weight = 0.8 + 0.2 = 1.0  ← keeps growing!

        # # With zero_grad():
        # Batch 1: gradient = 0.3  → zero_grad() → reset to 0
        # Batch 2: gradient = 0.5  → zero_grad() → reset to 0
        # Batch 3: gradient = 0.2  → zero_grad() → reset to 0
        # Without it your weight updates would be completely wrong — using accumulated gradients from ALL previous batches instead of just the current one.

        # In your PlantSnap training loop:
        # pythonfor imgs, labels in train_loader:     # loop through 219 batches
        #     optimizer.zero_grad()              # 🧹 clean slate for THIS batch
        #     outputs = model(imgs)              # forward pass
        #     loss = criterion(outputs, labels)  # calculate loss
        #     loss.backward()                    # backprop - calculate gradients
        #     optimizer.step()                   # update weights
        # Every single batch needs a clean start — otherwise batch 219's gradients would include baggage from all 218 previous batches! 😄

        # Interview answer:
        # "optimizer.zero_grad() clears gradients from the previous batch before computing new ones. 
        # PyTorch accumulates gradients by default, so without this call every batch would use corrupted 
        # gradients built up from all previous iterations, making training completely unstable."



        # Two completely different things:
        # Gradients    = temporary calculation tool (reset every batch) 🔄
        # Weights      = permanent memory of learning (NEVER reset)    💾

        # Analogy to make it crystal clear:
        # Think of a student doing math homework:
        # Weights   = knowledge in their brain (keeps growing forever)
        # Gradients = rough work on scratch paper (thrown away after each problem)

        # After each problem:
        # ✅ Answer gets written in brain (weights updated)
        # 🗑️ Scratch paper thrown away (gradients zeroed)

        # The student doesn't forget what they learned!
        # They just clear their working-out space for the next problem.

        # In your PlantSnap training:
        # Epoch 1, Batch 1:
        #   zero_grad()          🧹 clean scratch paper
        #   forward pass         📝 make prediction
        #   loss.backward()      🔍 calculate gradients (scratch work)
        #   optimizer.step()     🧠 UPDATE WEIGHTS permanently
        
        # Epoch 1, Batch 2:
        #   zero_grad()          🧹 clean scratch paper again
        #   forward pass         📝 make prediction (using UPDATED weights)
        #   loss.backward()      🔍 new gradients calculated fresh
        #   optimizer.step()     🧠 UPDATE WEIGHTS again permanently

        # The weights are what remember everything:
        # After Epoch 1:  weights adjusted 219 times ✅
        # After Epoch 2:  weights adjusted 219 more times ✅
        # After Epoch 10: weights adjusted 2,190 times total ✅

        # Gradients: reset 2,190 times (scratch paper cleared each time)
        # Weights:   never reset (brain keeps all learning permanently)

        # Visualizing weight improvement over time:
        # Start:    weight = 0.31  (random, knows nothing)
        # Batch 1:  weight = 0.28  (tiny improvement)
        # Batch 2:  weight = 0.25  (getting better)
        # ...
        # Batch 219: weight = 0.11 (end of epoch 1)
        # ...
        # Batch 2190: weight = 0.003 (end of training, well optimized!)
        # The gradient just told it which DIRECTION to nudge. The weight itself carries the accumulated knowledge forward permanently.

        # So to directly answer your question:

        # Gradients are NOT the training — they're just the compass pointing which way to go
        # Weights ARE the training — they permanently encode everything the model has learned
        # Clearing gradients = throwing away the compass after each step
        # The PATH you've walked (weight changes) is never forgotten


        # Interview answer for zero_grad:
        # "Gradients and weights are completely separate. Weights permanently accumulate learning across all batches and epochs
        # — they are never reset. Gradients are just temporary calculations used to figure out which direction 
        # to nudge each weight for the current batch. Once the weight has been updated via optimizer.step(), 
        # the gradient has done its job and must be cleared before the next batch to prevent stale calculations
        # corrupting the next update."
        optimizer.zero_grad()


        # FORWARD PASS: images flow through all 18 layers → predictions
        outputs = model(imgs)

        # imgs (batch of 16 herb images, each 224x224x3)
        #          ↓
        # Layer 1:  64 kernels scan every image → 64 feature maps
        #          ↓
        # Layer 2:  128 kernels scan feature maps → 128 feature maps  
        #          ↓
        # Layer 3-4: More convolutions → detecting textures
        #          ↓
        # Layer 5-8: Residual blocks → detecting shapes
        #          ↓
        # Layer 9-12: Residual blocks → detecting herb structures
        #          ↓
        # Layer 13-17: Complex combinations → chamomile-like patterns
        #          ↓
        # ❄️ All above = FROZEN (backbone doing its job silently)
        #          ↓
        # Layer 18 (model.fc): 512 features → 70 herb scores
        #          ↓
        # outputs = tensor of shape [16, 70]
        # 16 images × 70 confidence scores each

        #   outputs = [
        #   # Image 1 (chamomile):
        #   [0.12, 0.03, 2.41, -0.5, 0.8, ...]  # 70 raw scores
        
        #   # Image 2 (basil):
        #   [-0.3, 3.12, 0.04, 0.21, -0.9, ...] # 70 raw scores
        
        #   # ... 14 more images
        # ]

        # **Important — these are RAW scores not probabilities yet:**
        # ```
        # Raw scores can be anything: -2.3, 0.5, 3.1, -0.8
        # After softmax → probabilities between 0 and 1 that sum to 1.0
        # ```

        # CrossEntropyLoss handles the softmax conversion internally — that's why you don't need to call softmax separately during training. You only added it explicitly in your CoreML conversion code for iOS. Smart! ✅

        # ---

        # **The whole batch processes simultaneously:**
        # ```
        # Without GPU:  16 images processed one by one  🐢
        # With MPS GPU: 16 images processed in parallel 🚀

        #"outputs = model(imgs) runs the forward pass — all 16 herb images simultaneously flow through all 18 ResNet18 layers. The frozen backbone extracts 512 features per image capturing edges, textures, and herb structures. The final trainable layer maps those 512 features to 70 raw confidence scores. The output is a tensor of shape [16, 70] — 16 images each with 70 raw scores, one per herb class."

        #bit confused, we discussed that we are training on last 18th layer - then why we have to pass all the images through all the layers ?
        #         The confusion is completely understandable. Here's the key:
        # Training   = only happens in layer 18  ✅
        # Inference  = ALWAYS needs all 18 layers ✅
        # Both are true at the same time!

        # Think of it like a factory assembly line:
        # Layer 1-17: Workers who are EXPERTS (frozen, not learning)
        #              They still DO their job every single time
        #              They just don't CHANGE how they work

        # Layer 18:   The new apprentice (trainable, still learning)
        #              Learns from every batch
        #              But NEEDS the experts' output to do his job

        # Why layer 18 NEEDS all previous layers:
        # Raw herb image → Layer 18 directly?

        # Layer 18 only knows: "take 512 numbers, output 70 herb scores"
        # If you skip layers 1-17:
        #   Layer 18 receives raw pixels (224x224x3 = 150,528 numbers)
        #   It has NO IDEA what to do with raw pixels!
        #   It needs the 512 MEANINGFUL features the backbone prepared

        # What frozen actually means:
        # ❄️ FROZEN means:
        #   ✅ Still processes images every forward pass
        #   ✅ Still extracts edges, textures, shapes
        #   ❌ Does NOT update its weights via backprop
        #   ❌ Does NOT change how it processes images

        # 🔥 TRAINABLE means:
        #   ✅ Processes the 512 features from backbone
        #   ✅ Updates its weights every batch
        #   ✅ Learns which features = which herb

        # In your PlantSnap training loop:
        # Every batch of 16 herb images:

        # Forward pass (ALL 18 layers run):
        #   Layers 1-17: "Here are 512 meaningful herb features" ❄️
        #   Layer 18:    "Based on these features, it's chamomile" 🔥

        # Loss calculated:
        #   "Actually it was lavender, you were wrong"

        # Backward pass:
        #   Layer 18:    "I'll adjust my weights" 🔥 UPDATES
        #   Layers 1-17: "We see the error signal but ignore it" ❄️ NO UPDATE

        # Simple analogy:
        # Frozen backbone = experienced chef who preps ingredients
        # Final layer     = new chef learning to plate the dish

        # Every dish still needs:
        #   ✅ Experienced chef to prep ingredients (layers 1-17)
        #   ✅ New chef to plate it (layer 18)

        # But only the new chef is LEARNING how to improve.
        # The experienced chef already knows what he's doing.

        # The requires_grad = False connection:
        # pythonfor param in model.parameters():
        #     param.requires_grad = False  # ← this is what FREEZING means
        # This doesn't stop layers 1-17 from RUNNING — it just tells PyTorch:
        # "Don't bother calculating gradients for these layers during backprop"
        # They still do their job on every forward pass. They just don't learn or change.

        # Interview answer:
        # "All 18 layers always run during the forward pass — the frozen backbone still processes every image to extract meaningful features. Freezing means those layers don't UPDATE their weights during backpropagation, not that they stop working. Layer 18 needs the 512 features that layers 1-17 produce — without the backbone running, layer 18 would receive raw pixels it has no idea how to classify. The backbone does the heavy lifting of feature extraction, the trainable head learns which features correspond to which herb."

        # From where did we get 512 features 
        #512 is a design decision made by the ResNet18 creators at Microsoft Research in 2015. 🎯

        # `They designed ResNet18 with this specific architecture:
        # Input image (224x224x3)
        #     ↓
        # Conv Layer 1:    3   → 64  feature maps
        #     ↓
        # Residual Block:  64  → 64  feature maps
        #     ↓
        # Residual Block:  64  → 128 feature maps
        #     ↓
        # Residual Block:  128 → 256 feature maps
        #     ↓
        # Residual Block:  256 → 512 feature maps
        #     ↓
        # Average Pooling: 512 feature maps → 512 numbers  ← THIS is your 512
        #     ↓
        # Your fc layer:   512 → 70 herb classes

        # Why did they choose 512 specifically?
        # It was carefully engineered through experimentation:
        # Too few features (32)  → not enough information to classify well
        # Too many features (4096) → too slow, too much memory, overfitting
        # 512 → sweet spot for an 18-layer network ✅

        # The doubling pattern is intentional:
        # 64 → 128 → 256 → 512
        # Each block doubles the features while halving the spatial size:
        # Early layers:  Large image,  few features   (finding simple edges)
        # Later layers:  Small image,  many features  (finding complex patterns)
        # More features = richer description of what the model sees. By layer 17 the model has compressed your 224x224 herb image into 512 rich meaningful numbers.

        # Compare across ResNet family:
        # ResNet18:  ends at 512   ← yours
        # ResNet34:  ends at 512
        # ResNet50:  ends at 2048  (bigger bottleneck blocks)
        # ResNet101: ends at 2048
        # ResNet152: ends at 2048
        # That's why bigger ResNets need nn.Linear(2048, num_classes) instead of nn.Linear(512, num_classes).

        # Simple analogy:
        # 512 features = 512 questions the backbone answers about your herb image:

        # "How much green is there?"        → 0.82
        # "Are there circular shapes?"      → 0.91  (chamomile!)
        # "Is there a rough texture?"       → 0.23
        # "Are there thin stems visible?"   → 0.67
        # ... 508 more questions ...
        # Your final layer then learns:
        # "When questions 3, 7, 91 and 234 are all high → that's chamomile"

        # Interview answer:
        # "512 comes from ResNet18's architecture design by Microsoft Research. 
        # The network progressively doubles feature maps through residual blocks: 64 → 128 → 256 → 512,
        # while halving spatial dimensions at each stage. After the final average pooling layer, this produces
        # a 512-dimensional feature vector — a rich compressed summary of everything the backbone detected in the image. 
        # It's a carefully chosen sweet spot between too few features losing information and too many causing memory and overfitting issues."

        # from where 3 in (224X224X3) came from 
        #         The 3 = RGB channels of your herb image!
        # Every image has 3 channels:
        # R = Red   channel (224x224 pixels)
        # G = Green channel (224x224 pixels)  
        # B = Blue  channel (224x224 pixels)

        # Combined: 224 x 224 x 3

        # So the full input shape is:
        # Your batch of 16 herb images:
        # [16, 3, 224, 224]
        #   ↑   ↑   ↑    ↑
        #   │   │   │    └── width (pixels)
        #   │   │   └─────── height (pixels)
        #   │   └─────────── colour channels (RGB)
        #   └─────────────── batch size

        # Layer 1 takes 3 → 64 because:
        # Input:  3 channels  (R, G, B of your chamomile photo)
        # Output: 64 channels (64 different feature maps)

        # Each of the 64 kernels looks at ALL 3 colour channels simultaneously:
        #   Kernel 1: scans R+G+B together → detects horizontal edges
        #   Kernel 2: scans R+G+B together → detects vertical edges
        #   Kernel 3: scans R+G+B together → detects green patches
        #   ... 61 more kernels

        # Why this matters for herbs specifically:
        # Green channel → most important for herb identification
        #   High green + low red = fresh green leaf ✅
        #   High red + low green = dried/dead herb ❌

        # RGB together → lets kernels detect:
        #   Green stems vs brown stems
        #   Purple lavender vs yellow chamomile
        #   Red berries vs green leaves

        # If you had grayscale images:
        # Grayscale: [16, 1, 224, 224]  ← only 1 channel
        # Layer 1 would be: 1 → 64
        # You'd lose all colour information
        # Herb identification would be much harder!

        # The full picture now:
        # [16, 3, 224, 224]    ← your herb image batch
        #        ↓
        # Conv Layer 1: 3   → 64    ← 3 RGB channels in, 64 feature maps out
        #        ↓
        # Res Block 1:  64  → 64
        #        ↓
        # Res Block 2:  64  → 128
        #        ↓
        # Res Block 3:  128 → 256
        #        ↓
        # Res Block 4:  256 → 512
        #        ↓
        # Avg Pooling:  512 feature maps → 512 numbers
        #        ↓
        # Your fc:      512 → 70 herb classes
        # Every single number in that chain now has a reason! 💪

        # Interview answer:
        # "The 3 represents the RGB colour channels of the input image. 
        # Each herb photo is 224x224 pixels with 3 channels — red, green and blue. 
        # Layer 1 takes those 3 channels and produces 64 feature maps using 64 different kernels, 
        # each scanning all 3 colour channels simultaneously. 
        # For herb identification colour is particularly important — green channel variations help distinguish 
        # fresh vs dried herbs, and colour combinations help separate purple lavender from yellow chamomile."


        # 3 RGB channels in, 64 feature maps out - where is this 64 number comes from or how exact 64 kernels are selected
        # Same answer as 512 — Microsoft Research designed it that way! 😄

        # But let me explain the reasoning behind why 64 specifically:

        # It follows the same "sweet spot" logic:

        # Too few kernels (4)  → can't detect enough features → poor accuracy
        # Too many (256)       → too slow, too much memory, overkill for layer 1
        # 64                   → sweet spot for first layer ✅
        # Why first layer gets FEWER kernels than later layers:

        # Layer 1:  64  kernels  ← simple features (edges, colours)
        # Layer 2:  128 kernels  ← more complex (textures)
        # Layer 3:  256 kernels  ← even more complex (shapes)
        # Layer 4:  512 kernels  ← most complex (herb structures)
        # Early layers only need to detect SIMPLE things like edges and colour patches — you don't need 512 kernels to find horizontal lines.

        # Later layers combine those simple features into complex patterns — they need MORE kernels to capture all the combinations.

        # Think of it like language:

        # Layer 1 (64 kernels):   letters        (simple, few needed)
        # Layer 2 (128 kernels):  syllables      (more combinations)
        # Layer 3 (256 kernels):  words          (even more)
        # Layer 4 (512 kernels):  sentences      (most complex)
        # Final layer (70):       herb meanings  (your classification)
        # The honest truth about these numbers:

        # The ResNet authors didn't just GUESS 64. They ran hundreds of experiments:

        # Experiment 1: 32  → 64  → 128 → 256  → accuracy X
        # Experiment 2: 64  → 128 → 256 → 512  → accuracy Y ✅ winner
        # Experiment 3: 128 → 256 → 512 → 1024 → accuracy Z (too slow)
        # Then published the architecture that worked best. The whole field then adopted it as a standard.

        # This is called Architecture Search — finding the best combination of:

        # Number of layers
        # Number of kernels per layer
        # Kernel sizes
        # When to double channels
        # Modern networks like EfficientNet actually automate this process using algorithms to find optimal architectures rather than humans experimenting manually.

        # Interview answer:

        # "The 64 kernels in layer 1 was determined by the ResNet authors through empirical experimentation. The doubling pattern 64→128→256→512 follows a deliberate design principle — early layers detect simple features like edges and colours which don't require many kernels, while deeper layers combine those into complex patterns requiring more representational capacity. Too few kernels loses information, too many wastes computation. The ResNet paper established this as the optimal balance for an 18-layer network."

        # The key insight to remember:

        # These numbers are NOT magic or arbitrary.
        # They are ENGINEERED through experimentation.
        # And once proven to work → the whole field adopts them as standards.

        # That's why you'll see 64→128→256→512 in many CNN architectures
        # not just ResNet18 — it became a community standard.


        #than that way restnet should have only 4 layers with 512 reached
        #         You're absolutely right mathematically:
        # If doubling = the goal:
        # Layer 1: 64
        # Layer 2: 128
        # Layer 3: 256
        # Layer 4: 512  ← done! why 18 layers?
        # So why 18 layers?
        # Because the NUMBER of feature maps and the NUMBER of layers are two completely different things!

        # Each "layer" in ResNet18 is actually a RESIDUAL BLOCK containing multiple convolutions:
        # "Layer 1" is NOT just one convolution. It's:
        #   Conv 3x3 → BatchNorm → ReLU
        #   Conv 3x3 → BatchNorm → ReLU
        #   + Residual connection (skip connection)
        #   = ONE block but MULTIPLE operations

        # The 18 counts every single operation:
        # ResNet18 layer count:
        #   1 initial conv layer
        #   2 conv layers in block 1  (64 channels)
        #   2 conv layers in block 2  (64 channels)
        #   2 conv layers in block 3  (128 channels)
        #   2 conv layers in block 4  (128 channels)
        #   2 conv layers in block 5  (256 channels)
        #   2 conv layers in block 6  (256 channels)
        #   2 conv layers in block 7  (512 channels)
        #   2 conv layers in block 8  (512 channels)
        #   1 final fc layer
        #   ─────────────────
        #   = 18 layers total

        # Why multiple layers at the SAME channel size?
        # This is the key insight you discovered! 🎯
        # 64 channels → 64 channels (same size, 2 layers)
        # These layers are NOT changing the number of features — they are REFINING them!
        # First pass at 64:   "I can see some edges"
        # Second pass at 64:  "I can see BETTER edges, cleaner, more precise"

        # First pass at 128:  "I can see some textures"
        # Second pass at 128: "I can see BETTER textures, more detailed"
        # Think of it like sharpening a pencil drawing:
        # Rough sketch      → more detail → even more detail → final clean drawing
        # (first 64 layer)    (second 64)    (first 128)        (second 128)

        # Simple analogy:
        # Imagine learning to draw herbs:

        # Week 1: Learn basic shapes        (64 channels, pass 1)
        # Week 2: Refine those shapes       (64 channels, pass 2)
        # Week 3: Add texture detail        (128 channels, pass 1)
        # Week 4: Refine those textures     (128 channels, pass 2)
        # Week 5: Add complex structures    (256 channels, pass 1)
        # Week 6: Refine those structures   (256 channels, pass 2)
        # Week 7: Full herb recognition     (512 channels, pass 1)
        # Week 8: Perfect herb recognition  (512 channels, pass 2)
        # Each week you're getting BETTER at the same skill before moving to the next level.

        # Why not just jump straight to 512 in one layer?
        # One layer at 512:
        #   Model sees raw pixels → tries to understand everything at once
        #   Like reading War and Peace without knowing the alphabet first
        #   Result: terrible accuracy 💀

        # Multiple layers building up:
        #   Edges → textures → shapes → structures → herbs
        #   Each layer builds on the previous one
        #   Result: excellent accuracy ✅

        # This is called Hierarchical Feature Learning:
        # Layer 1-2:   pixels → edges           (alphabet)
        # Layer 3-4:   edges → textures         (words)
        # Layer 5-8:   textures → shapes        (sentences)
        # Layer 9-14:  shapes → structures      (paragraphs)
        # Layer 15-17: structures → herb parts  (chapters)
        # Layer 18:    herb parts → herb name   (full book understood)

        # Why ResNet18 vs ResNet50 vs ResNet152?
        # ResNet18:  18 layers  → 2 conv layers per block  → fast, good for mobile ✅ yours
        # ResNet50:  50 layers  → 3 conv layers per block  → more accurate, slower
        # ResNet152: 152 layers → 3 conv layers per block  → most accurate, slowest
        # More layers = more refinement = better accuracy BUT slower inference. For a mobile iOS app ResNet18 was the perfect choice!

        # Interview answer:
        # "The 18 in ResNet18 counts every individual convolution layer, not just the channel-doubling transitions. The architecture uses residual blocks containing multiple convolution layers at the same channel size before doubling. These same-size layers refine features rather than expand them — the first 64-channel layer detects rough edges, the second 64-channel layer sharpens those detections. This hierarchical refinement is what makes deep networks powerful. Simply jumping to 512 channels in one layer would be like trying to understand a sentence without first learning individual letters."
        outputs = model(imgs)

        # Calculate how wrong the predictions are
        # Single number: high = very wrong, low = nearly right


        # "If needed" — backpropagation ALWAYS happens every batch!
        # pythonloss = criterion(outputs, labels)  # Step 1: measure how wrong
        # loss.backward()                     # Step 2: ALWAYS backpropagate
        # optimizer.step()                    # Step 3: ALWAYS update weights
        # It's not conditional — every single batch goes through all three steps regardless of how good or bad the prediction was.

        # What this line actually does:
        # pythonoutputs = model(imgs)      # model's predictions  [16, 70] raw scores
        # labels  = actual herb IDs  # ground truth         [16] correct answers

        # loss = criterion(outputs, labels)
        # ```
        # ```
        # Image 1: model said basil (0.7)  but label = chamomile → BIG loss
        # Image 2: model said lavender (0.85) and label = lavender → tiny loss
        # Image 3: model said nettle (0.6)  but label = sage → BIG loss
        # ...all 16 images averaged into ONE loss number

        # What labels actually looks like:
        # pythonlabels = [2, 15, 43, 7, ...]  # just index numbers!
        # # 2  = chamomile (3rd in your CLASS_LABELS list)
        # # 15 = lavender
        # # 43 = sage
        # # 7  = basil
        # ```

        # CrossEntropyLoss compares:
        # ```
        # outputs[0] = [0.1, 0.05, 0.02, 0.7, ...]  ← model said index 3 (basil)
        # labels[0]  = 2                              ← correct answer is index 2 (chamomile)
        # Loss       = HIGH ❌ very wrong
        # ```

        # ---

        # **The loss number tells you everything about training health:**
        # ```
        # Epoch 1:  loss = 2.45  → model is guessing randomly
        # Epoch 3:  loss = 1.23  → starting to learn herb patterns
        # Epoch 6:  loss = 0.54  → getting good at common herbs
        # Epoch 10: loss = 0.18  → confidently identifying most herbs ✅

        # Interview answer:
        # "loss = criterion(outputs, labels) compares the model's 70 raw scores against the correct herb label for each of the 16 images in the batch, computing a single average loss number using CrossEntropyLoss. This loss represents how wrong the batch predictions were overall. It's not conditional — backpropagation always runs after this regardless of the loss value, using this number as the starting point to calculate how much each weight contributed to the error."

        #remind me how does it calculates the loss ? i mean without knowing that image is of chamomile or not - how do i know its wrong?
        #         You DO tell it the correct answer — that's what labels are!
        # Remember how you organized your training data:
        # plantsnap/herbs/train/
        #   ├── chamomile/
        #   │     ├── image1.jpg
        #   │     ├── image2.jpg
        #   │     └── image3.jpg
        #   ├── basil/
        #   │     ├── image1.jpg
        #   │     └── image2.jpg
        #   ├── lavender/
        #   │     └── image1.jpg

        # This line in your code did all the magic:
        # pythontrain_dataset = datasets.ImageFolder('plantsnap/herbs/train', transform=transform)
        # ```

        # `ImageFolder` automatically reads your folder structure and creates labels:
        # ```
        # chamomile/image1.jpg → label = 0
        # chamomile/image2.jpg → label = 0
        # basil/image1.jpg     → label = 1
        # basil/image2.jpg     → label = 1
        # lavender/image1.jpg  → label = 2
        # ```

        # **The folder name IS the label!** You labelled your data just by organizing it into folders! ✅

        # ---

        # **So during training:**
        # ```
        # DataLoader picks 16 images from folders:
        #   imgs   = [chamomile.jpg, basil.jpg, lavender.jpg ...]  ← actual images
        #   labels = [0,             1,         2            ...]  ← folder numbers

        # model predicts:
        #   outputs = [[high basil score], [high basil score], [high chamomile score]...]

        # CrossEntropyLoss compares:
        #   Image 1: predicted basil (1) but label says chamomile (0) → HIGH loss ❌
        #   Image 2: predicted basil (1) and label says basil (1)     → low loss  ✅
        #   Image 3: predicted chamomile (0) but label says lavender (2) → HIGH loss ❌
        # ```

        # ---

        # **This is called Supervised Learning:**
        # ```
        # Supervised   = YOU provide correct answers (folder labels)
        # Unsupervised = model finds patterns with NO labels
        # ```

        # Your PlantSnap is supervised learning — you supervised the model by organizing images into labelled folders.

        # ---

        # **The beautiful simplicity:**
        # ```
        # You organized 3,500 images into 70 folders
        # ImageFolder read those folder names as numbers
        # DataLoader paired each image with its folder number
        # CrossEntropyLoss compared predictions against those numbers
        # ```

        # That's it! Your folder organization was the entire labelling system! 😄

        # ---

        # **Interview answer:**

        # *"This is supervised learning. The correct answers come from the folder structure of the training data — ImageFolder automatically converts folder names into numeric labels. So chamomile images always carry label 2, basil images always carry label 15, and so on. CrossEntropyLoss then compares the model's predicted scores against these ground truth labels for each image in the batch, producing a single loss number representing average prediction error across all 16 images."*

        # ---

        # Now the full picture makes sense:
        # ```
        # You organized folders    → ImageFolder creates labels automatically
        # Labels travel with imgs  → DataLoader pairs them together
        # outputs vs labels        → CrossEntropyLoss calculates how wrong
        # loss.backward()          → backprop figures out who's responsible
        # optimizer.step()         → weights get nudged in right direction
        loss = criterion(outputs, labels)

        # BACKWARD PASS (Backpropagation!)
        # Error signal travels from layer 18 → layer 1 via chain rule
        # Each layer calculates its gradient (contribution to error)
        # Only unfrozen layers (model.fc) actually get gradients updated


        # Gradients tell weights WHICH WAY to change each batch → then get thrown away
        # Weights actually CHANGE each batch → and keep that change forever

        # In YOUR specific PlantSnap code — ONLY the final layer:
        # loss.backward():
        #   Layer 18 (fc):   gradient calculated ✅
        #   Layers 1-17:     gradient SKIPPED ⏭️ (requires_grad=False)

        # optimizer.step():
        #   Layer 18 (fc):   weight updated ✅
        #   Layers 1-17:     weight frozen ❄️ (never changes)

        # BUT — in a fully unfrozen network (no frozen backbone):
        # ALL layers would get:
        #   Gradients calculated ✅
        #   Weights updated      ✅
        # For example if you had trained ResNet18 from scratch:
        # Layer 1:  gradient calculated → weight updated every batch
        # Layer 2:  gradient calculated → weight updated every batch
        # ...
        # Layer 18: gradient calculated → weight updated every batch

        # So the answer depends on your architecture:
        # Your PlantSnap (frozen backbone):
        #   Gradients + weight updates = ONLY layer 18

        # Full fine-tuning (no freezing):
        #   Gradients + weight updates = ALL 18 layers

        # Partial freeze (last 4 layers unfrozen):
        #   Gradients + weight updates = layers 15, 16, 17, 18
        loss.backward()

        # GRADIENT DESCENT: update weights based on calculated gradients
        # Moves weights in direction that reduces loss
        # Step size controlled by learning rate (lr=0.001)

        # For PlantSnap:   updates ONLY layer 18 weights  ✅
        # In general: updates ALL layers' weights     ✅
        # Based on: gradients calculated by loss.backward() ✅
        optimizer.step()


        # Why .item() specifically?
        # pythonloss        = tensor(0.45, grad_fn=...)  # PyTorch tensor with gradient info
        # loss.item() = 0.45                       # just the plain Python number
        # Without .item() you'd be accumulating PyTorch tensors — wastes memory and keeps gradient history unnecessarily.
        # .item() extracts just the number you need.


        # "running_loss accumulates the loss from every batch in the epoch using loss.item() to extract the plain Python number from the PyTorch tensor.
        # After all 219 batches, dividing by the number of batches gives the average loss per batch. 
        # This average is more meaningful than the total because it stays consistent regardless of batch size, 
        # making it comparable across different experiments. The scheduler then watches this average to decide whether 
        # to reduce the learning rate."
        running_loss += loss.item()

        #     **Why average instead of total?**
        # ```
        # Total loss changes if you change batch size:
        #   batch_size=16  → 219 batches → total loss = 52.3
        #   batch_size=32  → 110 batches → total loss = 26.1  ← same training, different number!

        # Average loss stays consistent regardless of batch size:
        #   batch_size=16  → avg loss = 0.238
        #   batch_size=32  → avg loss = 0.238  ← same! ✅





    # Great — you spotted the full block! 🎯
    # Let's go through each line:

    # Line 1 — already covered:
    # pythonavg_loss = running_loss / len(train_loader)
    # Average loss across all 219 batches this epoch ✅

    # Line 2:
    # pythonold_lr = optimizer.param_groups[0]['lr']
    # ```

    # Saves the CURRENT learning rate BEFORE scheduler makes any changes:
    # ```
    # optimizer.param_groups[0]['lr'] = 0.001  ← current lr
    # old_lr = 0.001                           ← saved for comparison
    # ```

    # `param_groups[0]` — think of it as:
    # ```
    # optimizer keeps all parameters in groups:
    #   group 0 = model.fc.parameters()  ← your only group
    #   group 0 settings:
    #     lr     = 0.001
    #     weight = ...
    #     bias   = ...
    # You only have one group because you only passed model.fc.parameters() to Adam.

    # Line 3:
    # pythonscheduler.step(avg_loss)
    # ```

    # Scheduler looks at avg_loss and decides:
    # ```
    # Has loss improved in last 3 epochs? (patience=3)
    #   YES → keep lr the same
    #   NO  → multiply lr by 0.5 (factor=0.5)

    # Example:
    #   Epoch 7: avg_loss = 0.342
    #   Epoch 8: avg_loss = 0.341  ← barely improving
    #   Epoch 9: avg_loss = 0.340  ← barely improving
    #   Epoch 10: avg_loss = 0.340 ← no improvement for 3 epochs!
    
    #   Scheduler fires: lr = 0.001 × 0.5 = 0.0005 📉

    # Line 4:
    # pythonnew_lr = optimizer.param_groups[0]['lr']
    # ```

    # Reads the learning rate AFTER scheduler ran:
    # ```
    # If scheduler fired:   new_lr = 0.0005  (changed!)
    # If scheduler didn't:  new_lr = 0.001   (same)

    # Why save old_lr and new_lr?
    # Just for your print statement to show when lr changed:
    # pythonlr_note = f"  📉 LR → {new_lr:.6f}" if new_lr < old_lr else ""
    # print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}{lr_note}")
    # ```
    # ```
    # Epoch 7/10  - Loss: 0.3421
    # Epoch 8/10  - Loss: 0.3398
    # Epoch 9/10  - Loss: 0.3401
    # Epoch 10/10 - Loss: 0.3399  📉 LR → 0.000500
    # ```

    # That little 📉 tells you the scheduler kicked in — very useful for monitoring training!

    # ---

    # **The full scheduler story in your training:**
    # ```
    # Epochs 1-3:   loss dropping fast  → scheduler happy → lr stays 0.001
    # Epochs 4-6:   loss slowing down   → scheduler watching → lr stays 0.001
    # Epochs 7-9:   loss barely moving  → scheduler counting (patience=3)
    # Epoch 10:     loss stalled        → scheduler fires! → lr = 0.0005

    # Interview answer:
    # "This block monitors learning rate changes around the scheduler step. o
    # ld_lr captures the current rate before scheduler.step(avg_loss) evaluates whether loss has improved over the last 3 epochs. 
    # If loss hasn't improved, ReduceLROnPlateau halves the learning rate automatically. new_lr then captures the potentially updated rate. 
    # Comparing old_lr vs new_lr lets us print a notification when the scheduler fires, giving visibility into training dynamics. 
    # optimizer.param_groups[0] accesses the parameter group we passed to Adam — in our case just model.fc.parameters()
    avg_loss = running_loss / len(train_loader)   # ✅ compute once, use once
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_loss)                      # ✅ called once
    new_lr = optimizer.param_groups[0]['lr']
    lr_note = f"  📉 LR → {new_lr:.6f}" if new_lr < old_lr else ""
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}{lr_note}")



# Why .state_dict() and not just torch.save(model)?
# There are actually TWO ways to save in PyTorch:
# python# Option 1 - Save entire model (NOT recommended)
# torch.save(model, 'model.pth')

# # Option 2 - Save only weights (YOUR approach ✅ recommended)
# torch.save(model.state_dict(), 'my_herbs_model.pth')

# What state_dict actually contains:
# pythonstate_dict = {
#   'fc.weight': tensor([[0.003, 0.012, ...]]),  # 512x70 weights
#   'fc.bias':   tensor([0.001, 0.002, ...])      # 70 bias values
# }
# ```

# Just the learned numbers — nothing else! All 51,000 weights your final layer learned about herbs.

# ---

# **Why state_dict is better:**
# ```
# Save entire model:
#   Saves weights + architecture + PyTorch version
#   Breaks if PyTorch version changes ❌
#   Harder to modify architecture later ❌
#   Larger file size ❌

# Save state_dict:
#   Saves ONLY the learned weights ✅
#   Works across PyTorch versions ✅
#   Can load into modified architectures ✅
#   Smaller file size ✅
#   Industry standard ✅

# How you loaded it back in CoreML conversion:
# python# In your convert_coreml.py you did exactly this:
# backbone.load_state_dict(
#     torch.load('my_herbs_model.pth', map_location='cpu')
# )
# ```
# ```
# Save:  training   → state_dict → my_herbs_model.pth
# Load:  conversion → load_state_dict → CoreML → iOS
# ```

# The two files talk to each other perfectly! ✅

# ---

# **The .pth extension:**
# ```
# .pth = PyTorch file extension convention
# Could technically be .pt or anything
# .pth is the community standard

# Interview answer:
# "torch.save(model.state_dict()) saves only the learned weights rather than the entire model object. 
# state_dict is a dictionary mapping layer names to their weight tensors — in my case just fc.weight and fc.bias, 
# the 51,000 parameters my final layer learned. This is the recommended approach because it's portable across PyTorch versions 
# and smaller than saving the full model. In my CoreML conversion script I then loaded these weights back with load_state_dict() to reconstruct the model for iOS export."
torch.save(model.state_dict(), 'my_herbs_model.pth')
print("💾 SAVED: my_herbs_model.pth")
print("🎉 YOUR 4-HERB CLASSIFIER READY!")
