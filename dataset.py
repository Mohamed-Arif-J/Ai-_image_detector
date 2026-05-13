from datasets import load_dataset
import os
from PIL import Image
from tqdm import tqdm

# Load dataset
dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")

# Create output directories
os.makedirs("data/AiArtData", exist_ok=True)
os.makedirs("data/RealArt", exist_ok=True)

# Loop and save
for i, item in enumerate(tqdm(dataset, desc="Saving train data")):
    image = item["image"]
    label = item["label"]  # 0 = AI, 1 = Real

    # ✅ Convert non-RGB images safely
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ✅ Safer filename (use index instead of 'id')
    filename = f"{i:06d}.jpg"
    save_path = os.path.join("data/AiArtData" if label == 0 else "data/RealArt", filename)

    image.save(save_path, "JPEG")
