from datasets import load_dataset
dataset = load_dataset("beans")
dataset["train"].save_to_disk("plantsnap/train") 
dataset["validation"].save_to_disk("plantsnap/val")
print("✅ BEANS dataset ready! (10 classes, perfect for ResNet18 test)")
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")
