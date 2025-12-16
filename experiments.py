#!/usr/bin/env python3
"""
Experiments script for Textual Inversion with varying dataset sizes.
Trains TI models with 2, 4, 6, 10, 20, and 50 images from the impressionism dataset.
Each model is trained with dataset-size-appropriate training steps.
"""

import os
import shutil
import random
import json
import argparse
import subprocess
from pathlib import Path


# Configuration: image counts and corresponding training steps
EXPERIMENTS = [
    {"num_images": 2, "steps": 150},
    {"num_images": 4, "steps": 220},
    {"num_images": 6, "steps": 300},
    {"num_images": 10, "steps": 450},
    {"num_images": 20, "steps": 650},
    {"num_images": 50, "steps": 850},
]


def sample_images(source_dir, target_dir, num_images, seed=55):
    """
    Sample a subset of images from source_dir and copy to target_dir.
    
    Args:
        source_dir: Path to directory containing all images
        target_dir: Path to directory where sampled images will be copied
        num_images: Number of images to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled image filenames
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files
    all_images = [
        f for f in os.listdir(source_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if len(all_images) < num_images:
        raise ValueError(
            f"Not enough images in {source_dir}. "
            f"Found {len(all_images)}, requested {num_images}"
        )
    
    # Sample images
    random.seed(seed)
    sampled_images = random.sample(all_images, num_images)
    
    # Copy sampled images to target directory
    for img_name in sampled_images:
        src_path = os.path.join(source_dir, img_name)
        dst_path = os.path.join(target_dir, img_name)
        shutil.copy2(src_path, dst_path)
    
    print(f"Sampled {num_images} images to {target_dir}")
    return sampled_images


def train_textual_inversion(
    data_root,
    output_dir,
    max_steps,
    placeholder_token="<style1>",
    initializer_token="painting",
    learning_rate=5e-4,
    pretrained_model="runwayml/stable-diffusion-v1-5",
    what_to_teach="style",
    repeats=100,
    image_size=512,
    train_batch_size=1,
):
    """
    Train a textual inversion model using train_textual_inversion.py
    
    Args:
        data_root: Directory containing training images
        output_dir: Directory to save the trained model
        max_steps: Maximum training steps
        placeholder_token: Token to learn (e.g., <style1>)
        initializer_token: Token to initialize with
        learning_rate: Learning rate for training
        pretrained_model: Base model to use
        what_to_teach: "style" or "object"
        repeats: How many times to repeat the dataset
        image_size: Training image resolution
        train_batch_size: Batch size for training
    """
    cmd = [
        "python", "train_textual_inversion.py",
        "--data_root", data_root,
        "--output_dir", output_dir,
        "--placeholder_token", placeholder_token,
        "--initializer_token", initializer_token,
        "--max_train_steps", str(max_steps),
        "--learning_rate", str(learning_rate),
        "--pretrained_model_name_or_path", pretrained_model,
        "--what_to_teach", what_to_teach,
        "--train_batch_size", str(train_batch_size),
        "--repeats", str(repeats),
        "--image_size", str(image_size),
    ]
    
    print(f"\n{'='*80}")
    print(f"Training with {len(os.listdir(data_root))} images for {max_steps} steps")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    subprocess.run(cmd, check=True)


def save_experiment_metadata(
    output_dir,
    num_images,
    steps,
    sampled_images,
    data_root,
    source_dataset,
):
    """
    Save metadata about the experiment to the output directory.
    
    Args:
        output_dir: Directory where model was saved
        num_images: Number of images used for training
        steps: Number of training steps
        sampled_images: List of image filenames used
        data_root: Path to the training data directory
        source_dataset: Full path to the original dataset
    """
    metadata = {
        "num_images": num_images,
        "training_steps": steps,
        "sampled_images": sampled_images,
        "data_root": data_root,
        "source_dataset": source_dataset,
    }
    
    metadata_path = os.path.join(output_dir, "experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved experiment metadata to {metadata_path}")


def run_experiments(
    source_data_dir,
    base_output_dir,
    experiments=EXPERIMENTS,
    seed=55,
):
    """
    Run all experiments with different dataset sizes.
    
    Args:
        source_data_dir: Directory containing all impressionism images
        base_output_dir: Base directory for saving all experiment results
        experiments: List of experiment configurations
        seed: Random seed for reproducibility
    """
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create a summary file
    summary_path = os.path.join(base_output_dir, "experiments_summary.json")
    summary = {
        "source_dataset": source_data_dir,
        "base_output_dir": base_output_dir,
        "seed": seed,
        "experiments": [],
    }
    
    for exp in experiments:
        num_images = exp["num_images"]
        steps = exp["steps"]
        
        # Create experiment-specific directories
        exp_name = f"TI_impressionism_{num_images}imgs_{steps}steps"
        exp_data_dir = os.path.join(base_output_dir, exp_name, "dataset")
        exp_model_dir = os.path.join(base_output_dir, exp_name, "model")
        
        print(f"\n{'#'*80}")
        print(f"# Experiment: {exp_name}")
        print(f"{'#'*80}\n")
        
        # Sample images
        sampled_images = sample_images(
            source_dir=source_data_dir,
            target_dir=exp_data_dir,
            num_images=num_images,
            seed=seed,
        )
        
        # Train model
        train_textual_inversion(
            data_root=exp_data_dir,
            output_dir=exp_model_dir,
            max_steps=steps,
        )
        
        # Save experiment metadata
        save_experiment_metadata(
            output_dir=exp_model_dir,
            num_images=num_images,
            steps=steps,
            sampled_images=sampled_images,
            data_root=exp_data_dir,
            source_dataset=source_data_dir,
        )
        
        # Add to summary
        summary["experiments"].append({
            "name": exp_name,
            "num_images": num_images,
            "training_steps": steps,
            "data_dir": exp_data_dir,
            "model_dir": exp_model_dir,
        })
        
        print(f"\n✓ Completed experiment: {exp_name}\n")
    
    # Save summary
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Textual Inversion experiments with varying dataset sizes"
    )
    parser.add_argument(
        "--source_data_dir",
        type=str,
        default="artbench_data/impressionism",
        help="Directory containing all impressionism images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments_results",
        help="Base directory for saving all experiment results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for image sampling",
    )
    
    args = parser.parse_args()
    
    # Validate source directory exists
    if not os.path.isdir(args.source_data_dir):
        raise ValueError(f"Source data directory not found: {args.source_data_dir}")
    
    # Run all experiments
    run_experiments(
        source_data_dir=args.source_data_dir,
        base_output_dir=args.output_dir,
        experiments=EXPERIMENTS,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()