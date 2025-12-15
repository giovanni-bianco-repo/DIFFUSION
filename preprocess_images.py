# Preprocess images for textual inversion training
# This script resizes, optionally center-crops, and normalizes images in the dataset directory.

import os
from PIL import Image

# Settings
os.makedirs(output_dir, exist_ok=True)

# Refactored: support multiple style folders
input_root = "./inputs_textual_inversion"
output_root = "./inputs_textual_inversion_preprocessed"
image_size = 512  # Size expected by Stable Diffusion
center_crop = False  # Set True if you want center cropping

os.makedirs(output_root, exist_ok=True)

for style_name in os.listdir(input_root):
    style_path = os.path.join(input_root, style_name)
    if not os.path.isdir(style_path):
        continue
    out_style_path = os.path.join(output_root, style_name)
    os.makedirs(out_style_path, exist_ok=True)
    for fname in os.listdir(style_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(style_path, fname)
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if center_crop:
            w, h = img.size
            crop = min(w, h)
            left = (w - crop) // 2
            top = (h - crop) // 2
            img = img.crop((left, top, left + crop, top + crop))
        img = img.resize((image_size, image_size), resample=Image.BICUBIC)
        out_path = os.path.join(out_style_path, fname)
        img.save(out_path)
        print(f"Processed {style_name}/{fname} -> {out_path}")

print("All images preprocessed and saved to", output_root)
