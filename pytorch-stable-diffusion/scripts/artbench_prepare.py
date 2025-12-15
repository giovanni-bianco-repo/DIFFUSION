import wget
import tarfile
import os
from PIL import Image
from pathlib import Path

# Create directory
os.makedirs("impressionist_dataset", exist_ok=True)

# Download the dataset (256x256 version - good for LoRA/TI)
print("Downloading ArtBench-10...")
wget.download(
    url="https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar",
    out="artbench-10.tar"
)

# Extract
print("\nExtracting...")
with tarfile.open("artbench-10.tar") as tar:
    tar.extractall("artbench_data")

# Copy Cubism images (class label 2)
source_dir = Path("artbench_data/artbench-10-imagefolder-split/test/impressionism")
dest_dir = Path("impressionist_dataset")

count = 0
for img_file in source_dir.glob("*.jpg"):
    if count < 50:  # Get first 50 images
        img = Image.open(img_file)
        img.save(dest_dir / f"impressionism_{count:03d}.jpg")
        count += 1

print(f"\nSaved {count} Cubist images to cubist_dataset/")
