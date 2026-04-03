# ============================================================
# Herb Image Downloader & Preprocessor for Machine Learning
# Usage: pip install bing-image-downloader Pillow tqdm
# ============================================================

import os
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from bing_image_downloader import downloader

# ─── Configuration ───────────────────────────────────────────
HERBS = [
    # "lavender herb plant",
    # "basil herb plant",
    # "peppermint herb plant",
    # "rosemary herb plant",
    # "chamomile herb plant",
    # "turmeric herb plant",
    # "ginger root herb",
    # "echinacea herb plant",
    # "ashwagandha herb plant",
    # "holy basil tulsi herb",
    # "acanthaceae herb plant", 
    # "asian ginseng herb plant", #start 
    #"astragalus herb plant",
    #"birch herb plant", 
    #"black cohosh herb plant", 
    #"black haw herb plant", 
    # "black pepper herb plant", 
    # "black walnut herb plant",
    # "burdock herb plant",
    # "calendula herb plant", 
    # "california poppy herb plant", 
    # "catnip herb plant", 
    # "chaste tree herb tree", 
    #"chickweed herb plant", 
    # "comfrey herb plant", 
    # "coriander herb plant", 
    # "cramp bark herb plant", 
    # "cumin herb plant", 
    # "dandelion herb plant",
    # "elder berry herb plant", 
    # "elder berry flower herb plant",
    # "elecampane herb plant", 
    # "eleuthero herb plant", 
    # "fennel herb plant",
    # "feverfew herb plant", 
    # "garlic herb plant", 
    # "ginger herb plant",
    # "ginko leaf herb plant",
    # "green tea herb plant",
    # "Hops herb plant",
    # "Lady's mantle herb plant", 
    # "Lavendar herb plant", 
    # "Lemon balm herb plant", 
    # "Licorice root herb plant", 
    # "Linden herb plant", 
    # "Meadowsweet herb plant", 
    # "Motherwort herb plant", 
    # "Mullein herb plant", 
    # "Nettle herb plant", 
    # "Nutmeg herb plant",
    # "Oak herb plant", 
    # "Oat herb plant", 
    # "Orange herb plant", 
    # "Oregano herb plant", 
    # "Passionflower herb plant", 
    # "Plantain leaf herb plant", 
    # "Raspberry herb plant", 
    # "Red clover herb plant", 
    # "Reishi herb plant", 
    # "Rosemary herb plant", 
    # "Sage herb plant", 
    # "Saw palmetto herb plant", 
    # "sheperd's purse herb plant", 
    "shiitake herb plant", 
    "skullcap herb plant", 
    "splilanthes herb plant", 
    "st. john's wort herb plant", 
    "thyme herb plant", 
    "tulsi herb plant", 
    "valerian herb plant", 
    "vervain herb plant",
    "white pine herb plant", 
    "wild yam herb plant", 
    "willow herb plant", 
    "yarrow herb plant", 

]

IMAGES_PER_HERB   = 100
RAW_DIR           = Path("herbs_raw")        # downloaded originals
DATASET_DIR       = Path("herbs_dataset")    # cleaned & resized output
TARGET_SIZE       = (224, 224)               # ImageNet standard (224x224)
IMAGE_FORMAT      = "JPEG"
MIN_FILE_SIZE_KB  = 5                        # skip tiny/corrupt files

# ─── Step 1: Download ────────────────────────────────────────
def download_images():
    print("\n📥 Downloading herb images from Bing...\n")
    for herb in HERBS:
        safe_name = herb.replace(" ", "_")
        save_path = RAW_DIR / safe_name
        if save_path.exists() and len(list(save_path.glob("*"))) >= IMAGES_PER_HERB:
            print(f"  ⏭  Skipping '{herb}' — already downloaded.")
            continue
        print(f"  🌿 Downloading: {herb}")
        downloader.download(
            query=herb,
            limit=IMAGES_PER_HERB,
            output_dir=str(RAW_DIR),
            adult_filter_off=True,
            force_replace=False,
            timeout=10,
            verbose=False,
        )

# ─── Step 2: Clean & Resize ──────────────────────────────────
def preprocess_images():
    print("\n🔄 Preprocessing images (resize → 224×224, RGB)...\n")
    stats = {}

    for herb in HERBS:
        safe_name  = herb.replace(" ", "_")
        src_folder = RAW_DIR / safe_name
        dst_folder = DATASET_DIR / safe_name
        dst_folder.mkdir(parents=True, exist_ok=True)

        image_files = list(src_folder.glob("*"))
        saved, skipped = 0, 0

        for img_path in tqdm(image_files, desc=f"  {safe_name[:30]:<30}", ncols=80):
            # Skip very small files (likely corrupt or placeholder)
            if img_path.stat().st_size < MIN_FILE_SIZE_KB * 1024:
                skipped += 1
                continue

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")          # ensure 3-channel RGB
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)  # high-quality resize
                    out_name = f"{safe_name}_{saved + 1:04d}.jpg"
                    img.save(dst_folder / out_name, format=IMAGE_FORMAT, quality=95)
                    saved += 1
            except (UnidentifiedImageError, OSError):
                skipped += 1  # corrupt or unsupported format

        stats[safe_name] = {"saved": saved, "skipped": skipped}

    return stats

# ─── Step 3: Summary ─────────────────────────────────────────
def print_summary(stats):
    print("\n" + "═" * 50)
    print(f"  {'Herb':<35} {'Saved':>6}  {'Skipped':>7}")
    print("─" * 50)
    total_saved, total_skipped = 0, 0
    for herb, s in stats.items():
        print(f"  {herb[:35]:<35} {s['saved']:>6}  {s['skipped']:>7}")
        total_saved   += s["saved"]
        total_skipped += s["skipped"]
    print("═" * 50)
    print(f"  {'TOTAL':<35} {total_saved:>6}  {total_skipped:>7}")
    print(f"\n✅ Dataset saved to: ./{DATASET_DIR}/")
    print(f"📁 Structure: herbs_dataset/<herb_name>/<herb_name>_0001.jpg ...\n")

# ─── Step 4 (Optional): Verify Dataset ───────────────────────
def verify_dataset():
    """Quick sanity check — re-open every saved image."""
    print("🔍 Verifying saved images...\n")
    bad = []
    for img_path in DATASET_DIR.rglob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            bad.append(img_path)
            img_path.unlink()  # remove bad file

    if bad:
        print(f"  ⚠️  Removed {len(bad)} unreadable images during verification.")
    else:
        print("  ✅ All images passed verification.")

# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    download_images()
    stats = preprocess_images()
    print_summary(stats)
    verify_dataset()
