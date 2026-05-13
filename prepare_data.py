import os
import shutil
import random

# Paths to your datasets
AI_DIR = "data/AiArtData"
REAL_DIR = "data/RealArt"
BASE_OUTPUT = "data_split"

# Create train/val/test folders
for split in ["train", "val", "test"]:
    for cls in ["ai", "real"]:
        os.makedirs(os.path.join(BASE_OUTPUT, split, cls), exist_ok=True)

# Function to split and copy data
def split_and_copy(src_folder, dst_class, limit=None):
    files = os.listdir(src_folder)
    random.shuffle(files)

    # Optional: limit data for faster testing (set to None for full dataset)
    if limit:
        files = files[:limit]

    total = len(files)
    train_split = int(0.7 * total)
    val_split = int(0.15 * total)

    for i, f in enumerate(files):
        src_path = os.path.join(src_folder, f)
        if not os.path.isfile(src_path):
            continue

        if i < train_split:
            split = "train"
        elif i < train_split + val_split:
            split = "val"
        else:
            split = "test"

        dst_path = os.path.join(BASE_OUTPUT, split, dst_class, f)
        shutil.copy(src_path, dst_path)

    print(f"[✓] {dst_class.upper()} - {total} images split into train/val/test")

# --- Split both datasets ---
split_and_copy(AI_DIR, "ai")
split_and_copy(REAL_DIR, "real")

print("\n✅ Dataset prepared successfully and stored in 'data_split/'")
