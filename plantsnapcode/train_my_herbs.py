# =============================================================================
# PLANTSNAP HERB CLASSIFIER - ANNOTATED TRAINING CODE
# ResNet18 Transfer Learning | 70 Herb Classes | 3,461 Images
# =============================================================================
#
# TRAINING SUMMARY
# ─────────────────
# Model:    ResNet18 pretrained on ImageNet (1.2M images)
# Dataset:  3,461 herb images across 70 classes (50 images/class)
# Device:   Apple Silicon Mac (MPS backend)
# Output:   my_herbs_model.pth → converted to CoreML → iOS app (PlantSnap)
#
# PROBLEMS SOLVED DURING TRAINING
# ─────────────────────────────────
# Problem 1: NumPy Compatibility Error
#   Error:  RuntimeError: Numpy is not available
#   Cause:  NumPy 2.0 broke binary API that PyTorch was compiled against
#   Fix:    pip install "numpy<2"
#   Interview answer: "I hit a dependency conflict between NumPy 2.0 and
#   PyTorch. NumPy 2.0 broke the binary API PyTorch relied on. I downgraded
#   to NumPy 1.26.4 which resolved it."
#
# Problem 2: Loss Dropping Too Slowly
#   Observed: After 24 epochs loss was only at 3.98 (random baseline = ln(70) = 4.25)
#   Cause:    Learning rate too low → model taking tiny steps each epoch
#   Fix 1:    Increased epochs from 10 to 50
#   Fix 2:    Added ReduceLROnPlateau scheduler to auto-adapt learning rate
#
#   Why ln(70) = 4.25 is the random baseline:
#   With 70 equally likely classes, a random model assigns 1/70 probability
#   to each. Cross-entropy of uniform random guessing = -log(1/70) = ln(70) ≈ 4.25
#   So loss ABOVE 4.25 = worse than random. Loss BELOW 4.25 = actually learning!
#
#   Interview answer: "I trained ResNet18 on 3,461 herb images across 70 classes.
#   Initially loss was near the random baseline of ln(70)=4.25, meaning the model
#   was essentially guessing. After 24 epochs loss dropped to 3.98 but the pace
#   was too slow — about 0.013 per epoch. At that rate I'd need 100+ epochs to
#   reach meaningful accuracy. I increased epochs to 50 and added ReduceLROnPlateau
#   which halves the learning rate whenever improvement stalls for 3 consecutive
#   epochs. This handles both fast early learning and careful fine-tuning automatically."

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =============================================================================
# STEP 1: IMAGE PREPROCESSING PIPELINE
# =============================================================================
# Every herb image goes through these 4 steps before entering the network.
#
# Q: Why Resize(256) BEFORE CenterCrop(224)?
# A: Images come in all different sizes from the field:
#      Photo 1: 1200x800  (landscape DSLR)
#      Photo 2: 300x400   (small phone photo)
#      Photo 3: 4000x3000 (high-res camera)
#    If you CenterCrop(224) directly:
#      Photo 2 (300x400) → direct crop to 224 → barely anything left! ❌
#    With Resize(256) first:
#      ALL images standardized to same scale → THEN crop consistently ✅
#    Think of it like ironing fabric flat before cutting — never cut wrinkled fabric!
#
# Q: Why 224x224 specifically?
# A: ResNet18 was designed and trained on ImageNet at exactly 224x224.
#    Feeding a different size = wrong results. It's ResNet18's "native resolution."

transform = transforms.Compose([

    # Step 1: Resize shortest side to 256px
    # Standardizes all images to same scale regardless of original size.
    # Reduces computation while keeping aspect ratio intact.
    transforms.Resize(256),

    # Step 2: Crop center 224x224
    # Removes edge noise and artifacts (16px buffer on each side).
    # Keeps main subject — most herb photos have subject in center.
    # 224x224 = ResNet18's exact expected input size (native resolution).
    transforms.CenterCrop(224),

    # Step 3: Convert PIL image to PyTorch tensor
    # Automatically scales pixel values from [0, 255] to [0.0, 1.0].
    transforms.ToTensor(),

    # Step 4: Normalize using ImageNet mean and std values
    # [0.485, 0.456, 0.406] = mean RGB values of ALL 1.2M ImageNet images
    # [0.229, 0.224, 0.225] = std RGB values of ALL 1.2M ImageNet images
    # These are NOT gradient descent values — they are statistical properties
    # of the ImageNet dataset. ResNet18 was trained expecting images normalized
    # around these values — its "comfort zone" for pixel values.
    # Analogy: like making sure font size is always 12pt for someone who
    # learned to read in size 12. Feeding wrong scale = confusion.
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =============================================================================
# STEP 2: LOAD DATASET + DATALOADER
# =============================================================================
# ImageFolder reads folder structure and AUTOMATICALLY creates labels:
#   plantsnap/herbs/train/chamomile/image1.jpg → label = 0
#   plantsnap/herbs/train/basil/image1.jpg     → label = 1
#   plantsnap/herbs/train/lavender/image1.jpg  → label = 2
#
# The FOLDER NAME is the label — you labelled your data just by organizing
# images into folders! This is called SUPERVISED LEARNING:
#   Supervised   = YOU provide correct answers (via folder structure)
#   Unsupervised = model finds patterns with NO labels provided
#
# During training the DataLoader pairs each image with its folder number:
#   imgs   = [chamomile.jpg, basil.jpg, lavender.jpg ...]  ← actual images
#   labels = [0,             1,         2            ...]  ← folder numbers
# CrossEntropyLoss then compares model predictions against these folder numbers.
train_dataset = datasets.ImageFolder('plantsnap/herbs/train', transform=transform)

# Dataset stats:
#   50 images × 70 classes = 3,500 total images
#   3,500 ÷ batch_size 16  = 219 batches per epoch
#   219 batches × 10 epochs = 2,190 total weight updates
#
# Why 50 images/class is enough (barely):
#   < 50 images   → too little, model struggles 😬
#   50 images     → minimal but workable ✅  (BECAUSE we use transfer learning)
#   100-200       → comfortable zone 👍
#   1000+         → ideal 🎯
#   Transfer learning is what makes 50 images workable — ResNet18 already knows
#   edges, textures, shapes from ImageNet. We only teach it herb differences.

train_loader = DataLoader(
    train_dataset,

    # batch_size=16: Mini-batch gradient descent
    # Three flavors of gradient descent:
    #   Batch GD:     ALL 3,500 images at once → too slow, too much memory ❌
    #   Stochastic GD: ONE image at a time     → too noisy, unstable ❌
    #   Mini-batch GD: 16 images at a time     → best of both worlds ✅
    # Why 16 specifically: conservative for Mac MPS GPU memory.
    # Could try 32 if memory allows.
    batch_size=16,

    # shuffle=True: Randomize order every epoch
    # Without shuffle — model sees herbs in same order every epoch:
    #   Epoch 1: basil, basil, basil... chamomile, chamomile...
    #   Epoch 2: basil, basil, basil... (biased! thinks basil comes first)
    # With shuffle — random order every epoch:
    #   Epoch 1: chamomile, basil, nettle, lavender...
    #   Epoch 2: nettle, chamomile, basil, lavender... ✅ unbiased
    shuffle=True
)

print(f"✅ Found {len(train_dataset)} images across {len(train_dataset.classes)} classes")
print(f"Your classes: {train_dataset.classes}")


# =============================================================================
# STEP 3: DEVICE SETUP — USE MAC GPU (MPS)
# =============================================================================
# MPS = Metal Performance Shaders — Apple Silicon GPU acceleration.
# GPU processes entire batch of 16 images in PARALLEL (vs CPU one by one).
# Falls back to CPU automatically if MPS not available.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# =============================================================================
# STEP 4: MODEL SETUP — TRANSFER LEARNING WITH FROZEN BACKBONE
# =============================================================================
#
# WHAT IS TRANSFER LEARNING?
# ResNet18 was trained on 1.2M ImageNet images and already learned:
#   Layer 1-5:   edges, lines, colour patches        (letters)
#   Layer 6-10:  textures, patterns                  (words)
#   Layer 11-15: shapes, structures                  (sentences)
#   Layer 16-17: complex herb-like features           (paragraphs)
#   Layer 18:    "is this a cat/bird/car?" → replaced (we change this)
#
# WHY NOT TRAIN FROM SCRATCH?
# Training all 18 layers on only 50 images/class = catastrophic overfitting.
# Model would memorize training images and fail completely on new herbs.
# Transfer learning lets us borrow 1.2M image worth of knowledge.
#
# THE DOCTOR ANALOGY:
#   Frozen backbone = doctor keeps ALL medical school knowledge ❄️
#   Final layer     = retrain just their specialty (herbs instead of general medicine) 🔥
#   You wouldn't make them forget all of medical school — just retrain the specialty!

model = models.resnet18(weights='IMAGENET1K_V1')


# ❄️ FREEZE the backbone (layers 1-17)
# requires_grad=False tells PyTorch: "don't calculate gradients for these weights"
# IMPORTANT: Frozen does NOT mean layers stop working!
#   ✅ Frozen layers STILL run every forward pass (extract features)
#   ❌ Frozen layers do NOT update their weights during backpropagation
#   ❌ Frozen layers do NOT change how they process images
# Why all 18 layers run even though we only train layer 18:
#   Layer 18 only knows "take 512 numbers, output 70 herb scores"
#   If you skip layers 1-17, layer 18 receives raw pixels (150,528 numbers)
#   It has NO IDEA what to do with raw pixels — needs backbone's 512 features!
#   Think: experienced chef preps ingredients (layers 1-17), new chef plates dish (layer 18)
for param in model.parameters():
    param.requires_grad = False


# 🔥 Replace ONLY the final classification layer
# WHERE DOES 512 COME FROM?
# Microsoft Research designed ResNet18 with this specific architecture:
#   Input:          [3, 224, 224]     ← 3 = RGB channels (Red, Green, Blue)
#   Conv Layer 1:    3 → 64  maps    ← 3 RGB channels in, 64 feature maps out
#   Residual Block:  64 → 64  maps   ← REFINING edges (not expanding yet)
#   Residual Block:  64 → 128 maps   ← more complex textures
#   Residual Block: 128 → 256 maps   ← shapes and structures
#   Residual Block: 256 → 512 maps   ← complex herb patterns
#   Average Pooling: 512 maps → 512 numbers  ← THIS is your 512!
#   Your fc layer:   512 → 70 classes
#
# WHY THE DOUBLING PATTERN (64→128→256→512)?
# Early layers detect SIMPLE features (edges) — don't need many kernels.
# Later layers combine simple → complex — need MORE kernels for all combinations.
# Think like language: letters(64) → syllables(128) → words(256) → sentences(512)
#
# WHY 18 LAYERS AND NOT JUST 4 (one per channel-doubling)?
# Because each "layer" is a RESIDUAL BLOCK with MULTIPLE convolutions:
#   Conv 3x3 → BatchNorm → ReLU
#   Conv 3x3 → BatchNorm → ReLU
#   + skip connection
# The same-channel layers (64→64, 128→128) are REFINING not expanding:
#   First  64-layer: "I can see some edges"
#   Second 64-layer: "I can see BETTER, cleaner, more precise edges"
# Full count: 1 initial + 2×8 residual + 1 final fc = 18 layers
#
# WHY 64 KERNELS IN LAYER 1 SPECIFICALLY?
# Microsoft Research ran hundreds of experiments. Too few (4) = poor accuracy.
# Too many (256) = too slow for layer 1. 64 = proven sweet spot.
# The doubling pattern 64→128→256→512 became a community standard after the
# ResNet paper — you'll see it in many CNN architectures.
#
# WHERE DOES 3 COME FROM IN [3, 224, 224]?
# 3 = RGB colour channels. Every herb image has:
#   R channel (224×224 red values)
#   G channel (224×224 green values) ← most important for herbs!
#   B channel (224×224 blue values)
# High green + low red = fresh green leaf ✅  High red + low green = dried herb ❌
# Each of the 64 kernels in Layer 1 scans ALL 3 channels simultaneously.
# Grayscale [1, 224, 224] would lose all colour — herb ID would be much harder!
#
# THE 512 FEATURES = 512 QUESTIONS about your herb image:
#   "How much green is there?"       → 0.82
#   "Are there circular shapes?"     → 0.91  (chamomile!)
#   "Is there rough texture?"        → 0.23
#   "Are there thin stems visible?"  → 0.67
#   ... 508 more questions ...
# Your final layer learns: "when questions 3, 7, 91 and 234 are all high → chamomile"
#
# ResNet family feature sizes:
#   ResNet18:  512 features  ← yours (fast, perfect for iOS)
#   ResNet50:  2048 features (more accurate, slower)
#   ResNet152: 2048 features (most accurate, slowest)
model.fc = nn.Linear(512, len(train_dataset.classes))
model = model.to(device)


# =============================================================================
# STEP 5: LOSS FUNCTION, OPTIMIZER, SCHEDULER
# =============================================================================

# CrossEntropyLoss — measures HOW WRONG each prediction is
# "How wrong was my prediction, and by how much?"
#
# Example with chamomile image:
#   Good prediction: chamomile = 0.85 (85% confident) → small loss ✅
#   Bad prediction:  basil = 0.70, chamomile = 0.02   → BIG loss ❌
#   Large loss → big weight adjustment
#   Small loss → small adjustment
#   Zero loss  → no adjustment needed
#
# Why CrossEntropy for 70 herbs (not Binary Cross Entropy)?
#   Binary Cross Entropy → 2 classes (yes/no)
#   CrossEntropyLoss     → multiple classes (70 herbs) ✅
#
# Works together with F.softmax in CoreML conversion code:
#   Training:  CrossEntropyLoss handles softmax internally (don't call separately)
#   iOS/CoreML: F.softmax added explicitly to output probabilities for user display
criterion = nn.CrossEntropyLoss()


# Adam Optimizer — smarter gradient descent
#
# Adam = Adaptive Moment estimation
# Basic SGD: same learning rate for ALL 11M weights → one size fits all 😐
# Adam:      custom adaptive learning rate PER weight → each weight learns optimally 😎
#   Weights that haven't changed much → bigger nudge
#   Weights that change a lot         → smaller nudge (already moving fast)
#
# model.fc.parameters() NOT model.parameters() — WHY?
#   model.parameters()    → Adam loops through ALL 11M weights (wasteful)
#   model.fc.parameters() → Adam only looks at 51,000 weights (efficient) ✅
#   NOTE: Writing model.parameters() wouldn't BREAK training (frozen weights
#   have requires_grad=False so PyTorch skips them anyway) but it's wasteful —
#   Adam would iterate through 11M parameters to update only 51K (0.5%!).
#
# ResNet18 parameter breakdown:
#   Full backbone (layers 1-17): ~11,000,000 parameters  ← frozen, never updated
#   Your final layer (512×70+bias):     ~51,000 parameters  ← only these train
#   You trained just 0.5% of the network — that's why Mac training was fast!
#
# lr=0.001 — starting learning rate (step size going downhill)
#   Too high (0.1)    → overshoots minima, training goes crazy
#   Too low (0.00001) → trains but extremely slowly
#   0.001             → well-established sweet spot for Adam + transfer learning ✅
#
# HOW ALL COMPONENTS CONNECT:
#   CrossEntropyLoss  → measures HOW WRONG the prediction was
#   Backpropagation   → calculates WHICH weights caused the error
#   Adam optimizer    → decides HOW MUCH to adjust each weight
#   lr=0.001          → controls SIZE of each adjustment
#   ReduceLROnPlateau → shrinks lr automatically when progress stalls
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)


# ReduceLROnPlateau Scheduler — automatic learning rate decay
# Monitors avg_loss each epoch. If no improvement for 3 epochs → halves lr.
#
# Example progression:
#   Epochs 1-3:  loss dropping fast  → scheduler happy    → lr stays 0.001
#   Epochs 7-9:  loss barely moving  → scheduler counting (patience=3)
#   Epoch 10:    loss stalled        → scheduler fires!   → lr = 0.0005 📉
#   Epoch 14:    stalled again       → fires again!       → lr = 0.00025 📉
#
# mode='min':    watching for loss to DECREASE (not increase)
# factor=0.5:    cut learning rate in HALF when plateau detected
# patience=3:    wait 3 epochs before deciding it's truly a plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)


# =============================================================================
# STEP 6: TRAINING LOOP
# =============================================================================
#
# WHAT IS AN EPOCH?
# One complete pass through ALL 3,500 herb images.
# 10 epochs = model sees every herb 10 times in different random orders.
# When loss stops decreasing → model has converged → more epochs won't help much.
#
# HOW WEIGHTS AND GRADIENTS WORK (critical concept):
# ┌─────────────────────────────────────────────────────────┐
# │ GRADIENTS = temporary compass (thrown away each batch)  │
# │ WEIGHTS   = permanent memory  (NEVER reset, always grow)│
# └─────────────────────────────────────────────────────────┘
#
# Analogy: student doing homework
#   Weights   = knowledge in brain    (keeps growing forever)
#   Gradients = rough work on paper   (thrown away after each problem)
#   After each problem: answer written in brain ✅, scratch paper cleared 🗑️
#   Student doesn't forget what they learned! Just clears workspace.
#
# Weight improvement over 2,190 updates:
#   Start:       weight = 0.31  (random, knows nothing about herbs)
#   Batch 1:     weight = 0.28  (tiny improvement toward chamomile)
#   Batch 219:   weight = 0.11  (end of epoch 1, noticeably better)
#   Batch 2,190: weight = 0.003 (end of training, highly optimized!)
#
# FORWARD vs BACKWARD — both happen every batch:
#   Forward:  ALL 18 layers run     (even frozen ones extract features)
#   Backward: ONLY layer 18 trained (requires_grad=False skips layers 1-17)

NUM_EPOCHS = 10

print("🚀 Training YOUR herbs...")
for epoch in range(NUM_EPOCHS):
    model.train()   # Enables training-specific behaviour (dropout etc.)
    running_loss = 0

    for imgs, labels in train_loader:   # Each iteration = 1 mini-batch of 16 images
        imgs, labels = imgs.to(device), labels.to(device)

        # ── CLEAR GRADIENTS ──────────────────────────────────────────────────
        # PyTorch ACCUMULATES gradients by default — must clear before each batch!
        # Without zero_grad():
        #   Batch 1: gradient = 0.3
        #   Batch 2: gradient = 0.3 + 0.5 = 0.8  ← WRONG! accumulated
        #   Batch 3: gradient = 0.8 + 0.2 = 1.0  ← keeps growing forever!
        # With zero_grad() (correct):
        #   Batch 1: gradient = 0.3 → clear → 0
        #   Batch 2: gradient = 0.5 → clear → 0  ← each batch starts clean ✅
        # Every batch needs a clean start — otherwise batch 219 would carry
        # garbage gradients from all 218 previous batches!
        optimizer.zero_grad()

        # ── FORWARD PASS ─────────────────────────────────────────────────────
        # All 16 herb images simultaneously flow through all 18 layers:
        #   [16, 3, 224, 224]     ← batch of 16 RGB herb images
        #         ↓
        #   Layer 1 (3→64):   detects edges, colour patches  ❄️ frozen
        #         ↓
        #   Layers 2-4:       textures, patterns             ❄️ frozen
        #         ↓
        #   Layers 5-12:      shapes, herb structures        ❄️ frozen
        #         ↓
        #   Layers 13-17:     complex herb combinations      ❄️ frozen
        #         ↓
        #   Layer 18 (fc):    512 features → 70 herb scores  🔥 trainable
        #         ↓
        #   outputs shape: [16, 70]  ← 16 images × 70 raw confidence scores
        #
        # These are RAW scores (can be negative): [-2.3, 0.5, 3.1, -0.8...]
        # NOT probabilities yet! CrossEntropyLoss handles softmax internally.
        # (You added explicit softmax only in CoreML conversion for iOS display)
        #
        # Q: Why run all 18 layers if we only TRAIN layer 18?
        # A: Training = only layer 18. But layer 18 NEEDS layers 1-17's output!
        #    Layer 18 only knows "take 512 numbers → output 70 herb scores"
        #    Without backbone: layer 18 gets raw pixels (150,528 numbers) → useless!
        #    Frozen = still runs every pass, just never changes its weights.
        outputs = model(imgs)

        # ── CALCULATE LOSS ───────────────────────────────────────────────────
        # Compares model's 70 raw scores against correct herb label for each image.
        # labels = [2, 15, 43, 7...] — just index numbers from ImageFolder folder names
        #
        # Example for one batch image:
        #   outputs[0] = [0.1, 0.05, 0.02, 0.7, ...]  ← model predicted basil (index 3)
        #   labels[0]  = 2                              ← correct answer is chamomile (index 2)
        #   loss = HIGH ❌ very wrong
        #
        # All 16 images averaged into ONE loss number.
        # Backpropagation ALWAYS runs after this — not conditional on loss size!
        #
        # Training health check — what loss values mean:
        #   loss > 4.25 → worse than random guessing (ln 70 = 4.25 baseline)
        #   loss ~2.45  → model guessing randomly but improving
        #   loss ~1.23  → starting to learn herb patterns
        #   loss ~0.54  → getting good at common herbs
        #   loss ~0.18  → confidently identifying most herbs ✅
        loss = criterion(outputs, labels)

        # ── BACKPROPAGATION ──────────────────────────────────────────────────
        # Error signal travels backwards through layers using the chain rule.
        #
        # Gradients tell weights WHICH WAY to change → then get thrown away
        # Weights actually CHANGE each batch → and keep that change forever
        #
        # Because requires_grad=False on backbone:
        #   Layer 18: gradient calculated ✅ weight will update ✅
        #   Layers 1-17: gradient SKIPPED ⏭️ (PyTorch doesn't even calculate them)
        #
        # This is also why frozen backbone is computationally EFFICIENT —
        # backprop stops early instead of flowing through all 18 layers!
        #
        # For fully unfrozen network (no freezing):
        #   ALL layers → gradients calculated → weights updated every batch
        #
        # For your PlantSnap (frozen backbone):
        #   ONLY layer 18 → gradients calculated → weights updated
        loss.backward()

        # ── UPDATE WEIGHTS (GRADIENT DESCENT) ────────────────────────────────
        # Adam uses the calculated gradients to nudge weights downhill.
        # Only updates model.fc.parameters() — the 51,000 weights we passed to Adam.
        # Layers 1-17 weights: never touched ❄️
        # Layer 18 weights:    permanently updated with each call 🔥
        #
        # PlantSnap:  updates ONLY layer 18 weights ✅
        # General:    updates ALL layers' weights ✅ (when nothing is frozen)
        optimizer.step()

        # ── ACCUMULATE BATCH LOSS ────────────────────────────────────────────
        # .item() extracts plain Python number from PyTorch tensor.
        # Without .item(): accumulating tensors wastes memory + keeps gradient history.
        # running_loss builds up the total loss across all 219 batches this epoch.
        running_loss += loss.item()


    # ── END OF EPOCH: CALCULATE AVERAGE LOSS ─────────────────────────────────
    # Average loss per batch — more meaningful than total loss because it stays
    # consistent regardless of batch_size:
    #   batch_size=16 → 219 batches → total loss = 52.3, avg = 0.238
    #   batch_size=32 → 110 batches → total loss = 26.1, avg = 0.238 ← same! ✅
    # Average makes metrics comparable across experiments.
    avg_loss = running_loss / len(train_loader)

    # ── SAVE LEARNING RATE BEFORE SCHEDULER ──────────────────────────────────
    # optimizer.param_groups[0] = the group of parameters we passed to Adam
    # (just model.fc.parameters() — only one group in our case)
    # Saved BEFORE scheduler runs so we can detect if it changed.
    old_lr = optimizer.param_groups[0]['lr']

    # ── SCHEDULER STEP ───────────────────────────────────────────────────────
    # Scheduler checks: has avg_loss improved in the last 3 epochs (patience=3)?
    #   YES → keep lr the same
    #   NO  → multiply lr by 0.5 (factor=0.5)
    # Called ONCE per epoch (not per batch) — watching epoch-level trends.
    scheduler.step(avg_loss)

    # ── DETECT IF LR CHANGED ─────────────────────────────────────────────────
    # Compare new lr against old lr to show 📉 notification when scheduler fires.
    new_lr = optimizer.param_groups[0]['lr']
    lr_note = f"  📉 LR → {new_lr:.6f}" if new_lr < old_lr else ""
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}{lr_note}")


# =============================================================================
# STEP 7: SAVE THE TRAINED MODEL
# =============================================================================
# TWO ways to save in PyTorch:
#
#   Option 1 — Save entire model (NOT recommended):
#     torch.save(model, 'model.pth')
#     Saves weights + architecture + PyTorch version → breaks on version changes ❌
#
#   Option 2 — Save only weights via state_dict (YOUR approach ✅):
#     torch.save(model.state_dict(), 'my_herbs_model.pth')
#     Saves ONLY the learned weights → portable, smaller, industry standard ✅
#
# What state_dict contains:
#   {
#     'fc.weight': tensor([[0.003, 0.012, ...]]),  # 512×70 = 35,840 weights
#     'fc.bias':   tensor([0.001, 0.002, ...])      # 70 bias values
#   }
#   Just 51,000 numbers — everything your final layer learned about 70 herbs.
#
# How the two files connect:
#   train_herbs.py:     training → state_dict → my_herbs_model.pth    (saved here)
#   convert_coreml.py:  load_state_dict('my_herbs_model.pth') → CoreML → iOS  (loaded there)
#
# The .pth extension is PyTorch community convention (could technically be .pt).
torch.save(model.state_dict(), 'my_herbs_model.pth')
print("💾 SAVED: my_herbs_model.pth")
print("🎉 YOUR 70-HERB CLASSIFIER READY!")