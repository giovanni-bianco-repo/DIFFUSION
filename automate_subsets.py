import os
import shutil
import random
import subprocess
from pathlib import Path

# Settings
full_data_dir = "./inputs_textual_inversion_preprocessed/style1"
subsets_root = "./subsets"
results_root = "./subset_results"
clip_scores_root = "./clip_scores"
num_runs = 3  # Number of times to train/test per subset
subset_sizes = [2, 3, 4]  # Example: try with 2, 3, 4 images per subset

os.makedirs(subsets_root, exist_ok=True)
os.makedirs(results_root, exist_ok=True)
os.makedirs(clip_scores_root, exist_ok=True)

all_images = [f for f in os.listdir(full_data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for subset_size in subset_sizes:
    for run in range(num_runs):
        subset_name = f"subset_{subset_size}_run_{run+1}"
        subset_dir = os.path.join(subsets_root, subset_name)
        os.makedirs(subset_dir, exist_ok=True)
        # Randomly sample images for this subset
        subset_imgs = random.sample(all_images, min(subset_size, len(all_images)))
        for img in subset_imgs:
            shutil.copy(os.path.join(full_data_dir, img), os.path.join(subset_dir, img))
        # Train
        train_dir = f"./style1-output-{subset_name}"
        train_cmd = [
            "python3", "train_textual_inversion.py",
            # You may need to add arguments to your train script to accept data/output dirs
        ]
        # You must edit train_textual_inversion.py to accept data/output dir as arguments for this to work
        print(f"Training on {subset_name}...")
        #subprocess.run(train_cmd)  # Uncomment after editing train_textual_inversion.py
        # Test
        test_cmd = [
            "python3", "test_textual_inversion.py",
            # You may need to add arguments to your test script to accept model dir/output dir
        ]
        print(f"Testing on {subset_name}...")
        #subprocess.run(test_cmd)  # Uncomment after editing test_textual_inversion.py
        # Save results (images, clip scores)
        # You must edit your test script to save outputs to results_root and clip_scores_root
        # Example: shutil.move("cubism_trained_output.png", os.path.join(results_root, f"{subset_name}_trained.png"))
        # Example: shutil.move("clip_scores.txt", os.path.join(clip_scores_root, f"{subset_name}_clip_scores.txt"))

print("Done. Please ensure your train/test scripts accept data/output arguments for full automation.")
