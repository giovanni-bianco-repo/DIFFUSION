#!/usr/bin/env python3
"""
Comprehensive testing script for Textual Inversion experiments.
Generates images with base model and all trained models, computes metrics,
and creates visualizations showing how performance scales with dataset size.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
import clip
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel


# Test prompts
TEST_PROMPTS = [
    "a small red airplane flying over mountains in <style1> style",
    "three people walking on a busy city stret in <style1> style",
    "a cute cat sitting on a wooden table in <style1> style",
]

# Corresponding prompts for base model (without placeholder token)
BASE_PROMPTS = [
    "a small red airplane flying over mountains in style1 style",
    "three people walking on a busy city stret in style1 style",
    "a cute cat sitting on a wooden table in style1 style",
]


class MetricsCalculator:
    """Calculate various metrics for generated images."""
    
    def __init__(self, device="cuda"):
        self.device = device
        
        # Load CLIP model for text-image similarity
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Load CLIP for feature extraction
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_feature_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        
        # Image transform for style consistency
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def clip_text_similarity(self, image, text):
        """
        Calculate CLIP similarity between image and text prompt.
        Higher is better (range: 0-1, typically 0.2-0.35 for good matches).
        """
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
        
        return similarity
    
    def style_consistency_with_training_data(self, generated_image, training_images_dir):
        """
        Calculate style consistency between generated image and training images.
        Uses CLIP image embeddings to measure visual similarity.
        Returns average cosine similarity with training images.
        """
        # Get generated image features
        gen_img = self.transform(generated_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            gen_inputs = self.clip_processor(images=generated_image, return_tensors="pt").to(self.device)
            gen_features = self.clip_feature_model.get_image_features(**gen_inputs)
            gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
        
        # Load and process training images
        training_images = [
            f for f in os.listdir(training_images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        similarities = []
        for img_file in training_images[:20]:  # Sample max 20 images for efficiency
            img_path = os.path.join(training_images_dir, img_file)
            try:
                train_img = Image.open(img_path).convert('RGB')
                
                with torch.no_grad():
                    train_inputs = self.clip_processor(images=train_img, return_tensors="pt").to(self.device)
                    train_features = self.clip_feature_model.get_image_features(**train_inputs)
                    train_features = train_features / train_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (gen_features @ train_features.T).item()
                    similarities.append(similarity)
            except Exception as e:
                print(f"Warning: Could not process {img_file}: {e}")
                continue
        
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_diversity(self, images):
        """
        Calculate diversity among generated images.
        Lower similarity = higher diversity.
        """
        if len(images) < 2:
            return 0.0
        
        features = []
        for img in images:
            with torch.no_grad():
                inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
                feats = self.clip_feature_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                features.append(feats)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                sim = (features[i] @ features[j].T).item()
                similarities.append(sim)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def image_quality_score(self, image):
        """
        Estimate image quality using variance and sharpness.
        Simple heuristic: higher variance and edge strength = better quality.
        """
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Variance (higher = more detail)
        variance = np.var(img_array)
        
        # Edge strength using Sobel-like operator
        dx = np.abs(np.diff(img_array, axis=1))
        dy = np.abs(np.diff(img_array, axis=0))
        edge_strength = np.mean(dx) + np.mean(dy)
        
        # Normalize and combine
        quality = (variance / 10000.0 + edge_strength / 100.0) / 2.0
        return min(quality, 1.0)


def load_experiment_info(experiments_dir):
    """Load information about all experiments from the summary file."""
    summary_path = os.path.join(experiments_dir, "experiments_summary.json")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Experiments summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return summary


def generate_images_for_model(
    pipeline,
    prompts,
    output_dir,
    model_name,
    num_images_per_prompt=5,
    seed=42,
):
    """
    Generate images using a given pipeline for all prompts.
    
    Returns:
        Dictionary mapping prompt to list of generated images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    results = {}
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"  Generating images for prompt {prompt_idx + 1}/{len(prompts)}: '{prompt}'")
        
        images = []
        for img_idx in range(num_images_per_prompt):
            # Generate image
            output = pipeline(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
            )
            image = output.images[0]
            
            # Save image
            filename = f"{model_name}_prompt{prompt_idx}_img{img_idx}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            
            images.append(image)
        
        results[prompt] = images
    
    return results


def compute_metrics_for_images(
    images_dict,
    prompts,
    training_data_dir,
    metrics_calculator,
    model_name,
):
    """
    Compute all metrics for generated images.
    
    Returns:
        Dictionary with aggregated metrics
    """
    all_metrics = []
    
    for prompt_idx, (prompt, images) in enumerate(images_dict.items()):
        # Text-to-style prompt for CLIP (remove placeholder token for evaluation)
        eval_prompt = prompts[prompt_idx] if isinstance(prompts, list) else prompt
        
        for img_idx, image in enumerate(images):
            metrics = {
                'model': model_name,
                'prompt_idx': prompt_idx,
                'prompt': prompt,
                'image_idx': img_idx,
            }
            
            # CLIP text similarity
            metrics['clip_text_similarity'] = metrics_calculator.clip_text_similarity(
                image, eval_prompt
            )
            
            # Style consistency with training data
            if training_data_dir and os.path.exists(training_data_dir):
                metrics['style_consistency'] = metrics_calculator.style_consistency_with_training_data(
                    image, training_data_dir
                )
            else:
                metrics['style_consistency'] = 0.0
            
            # Image quality
            metrics['image_quality'] = metrics_calculator.image_quality_score(image)
            
            all_metrics.append(metrics)
        
        # Calculate diversity for this prompt
        diversity = metrics_calculator.calculate_diversity(images)
        for m in all_metrics[-len(images):]:
            m['diversity'] = diversity
    
    return all_metrics


def create_visualizations(results_df, output_dir):
    """Create plots showing how metrics evolve with dataset size."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Extract dataset size from model names
    def get_num_images(model_name):
        if 'base' in model_name.lower():
            return 0
        parts = model_name.split('_')
        for part in parts:
            if 'imgs' in part:
                return int(part.replace('imgs', ''))
        return 0
    
    results_df['num_training_images'] = results_df['model'].apply(get_num_images)
    
    # Aggregate metrics by model
    model_metrics = results_df.groupby(['model', 'num_training_images']).agg({
        'clip_text_similarity': ['mean', 'std'],
        'style_consistency': ['mean', 'std'],
        'image_quality': ['mean', 'std'],
        'diversity': 'mean',
    }).reset_index()
    
    model_metrics.columns = [
        'model', 'num_training_images',
        'clip_text_similarity_mean', 'clip_text_similarity_std',
        'style_consistency_mean', 'style_consistency_std',
        'image_quality_mean', 'image_quality_std',
        'diversity_mean'
    ]
    
    # Sort by number of training images
    model_metrics = model_metrics.sort_values('num_training_images')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Textual Inversion Performance vs Dataset Size', fontsize=16, fontweight='bold')
    
    # Plot 1: CLIP Text Similarity
    ax = axes[0, 0]
    ax.errorbar(
        model_metrics['num_training_images'],
        model_metrics['clip_text_similarity_mean'],
        yerr=model_metrics['clip_text_similarity_std'],
        marker='o', linewidth=2, markersize=8, capsize=5
    )
    ax.set_xlabel('Number of Training Images', fontsize=12)
    ax.set_ylabel('CLIP Text Similarity', fontsize=12)
    ax.set_title('Prompt Adherence (CLIP Score)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Style Consistency
    ax = axes[0, 1]
    trained_models = model_metrics[model_metrics['num_training_images'] > 0]
    ax.errorbar(
        trained_models['num_training_images'],
        trained_models['style_consistency_mean'],
        yerr=trained_models['style_consistency_std'],
        marker='s', linewidth=2, markersize=8, capsize=5, color='green'
    )
    ax.set_xlabel('Number of Training Images', fontsize=12)
    ax.set_ylabel('Style Consistency Score', fontsize=12)
    ax.set_title('Style Similarity with Training Data', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Image Quality
    ax = axes[1, 0]
    ax.errorbar(
        model_metrics['num_training_images'],
        model_metrics['image_quality_mean'],
        yerr=model_metrics['image_quality_std'],
        marker='^', linewidth=2, markersize=8, capsize=5, color='orange'
    )
    ax.set_xlabel('Number of Training Images', fontsize=12)
    ax.set_ylabel('Image Quality Score', fontsize=12)
    ax.set_title('Generated Image Quality', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Diversity
    ax = axes[1, 1]
    ax.plot(
        model_metrics['num_training_images'],
        model_metrics['diversity_mean'],
        marker='D', linewidth=2, markersize=8, color='purple'
    )
    ax.set_xlabel('Number of Training Images', fontsize=12)
    ax.set_ylabel('Diversity Score', fontsize=12)
    ax.set_title('Output Diversity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'metrics_vs_dataset_size.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved metrics plot: {plot_path}")
    
    # Create a detailed comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Detailed Metrics Comparison Across All Models', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('clip_text_similarity_mean', 'CLIP Text Similarity', 'blue'),
        ('style_consistency_mean', 'Style Consistency', 'green'),
        ('image_quality_mean', 'Image Quality', 'orange'),
    ]
    
    for idx, (metric, title, color) in enumerate(metrics_to_plot):
        ax = axes[idx]
        if metric == 'style_consistency_mean':
            data = model_metrics[model_metrics['num_training_images'] > 0]
        else:
            data = model_metrics
        
        ax.bar(range(len(data)), data[metric], color=color, alpha=0.7)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([f"{int(n)} imgs" if n > 0 else "base" for n in data['num_training_images']], 
                           rotation=45, ha='right')
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save detailed plot
    detailed_plot_path = os.path.join(output_dir, 'detailed_metrics_comparison.png')
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed comparison plot: {detailed_plot_path}")
    
    plt.close('all')
    
    # Save model metrics table
    metrics_table_path = os.path.join(output_dir, 'model_metrics_summary.csv')
    model_metrics.to_csv(metrics_table_path, index=False)
    print(f"Saved metrics summary: {metrics_table_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Textual Inversion experiments and generate comparison metrics"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments_results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Directory to save test results and visualizations",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=5,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip image generation and only compute metrics/plots from existing images",
    )
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load experiment information
    print("\nLoading experiment information...")
    experiments_info = load_experiment_info(args.experiments_dir)
    
    # Initialize metrics calculator
    print("\nInitializing metrics calculator...")
    metrics_calc = MetricsCalculator(device=device)
    
    all_results = []
    
    if not args.skip_generation:
        # Generate images with base model
        print("\n" + "="*80)
        print("TESTING BASE MODEL")
        print("="*80)
        
        base_pipe = StableDiffusionPipeline.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        
        base_output_dir = os.path.join(args.output_dir, "images", "base_model")
        base_images = generate_images_for_model(
            base_pipe,
            BASE_PROMPTS,
            base_output_dir,
            "base_model",
            num_images_per_prompt=args.num_images_per_prompt,
            seed=args.seed,
        )
        
        # Compute metrics for base model
        print("\nComputing metrics for base model...")
        base_metrics = compute_metrics_for_images(
            base_images,
            BASE_PROMPTS,
            None,  # No training data for base model
            metrics_calc,
            "base_model",
        )
        all_results.extend(base_metrics)
        
        # Clean up base model
        del base_pipe
        torch.cuda.empty_cache()
        
        # Test each trained model
        for exp in experiments_info['experiments']:
            exp_name = exp['name']
            model_dir = exp['model_dir']
            data_dir = exp['data_dir']
            num_images = exp['num_images']
            
            print("\n" + "="*80)
            print(f"TESTING: {exp_name}")
            print("="*80)
            
            # Load trained model
            print(f"Loading model from {model_dir}...")
            try:
                trained_pipe = StableDiffusionPipeline.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                ).to(device)
            except Exception as e:
                print(f"Error loading model {exp_name}: {e}")
                continue
            
            # Generate images
            model_output_dir = os.path.join(args.output_dir, "images", exp_name)
            trained_images = generate_images_for_model(
                trained_pipe,
                TEST_PROMPTS,
                model_output_dir,
                exp_name,
                num_images_per_prompt=args.num_images_per_prompt,
                seed=args.seed,
            )
            
            # Compute metrics
            print(f"\nComputing metrics for {exp_name}...")
            trained_metrics = compute_metrics_for_images(
                trained_images,
                BASE_PROMPTS,  # Use base prompts for evaluation
                data_dir,
                metrics_calc,
                exp_name,
            )
            all_results.extend(trained_metrics)
            
            # Clean up
            del trained_pipe
            torch.cuda.empty_cache()
    
        # Save all metrics
        results_df = pd.DataFrame(all_results)
        metrics_path = os.path.join(args.output_dir, "all_metrics.csv")
        results_df.to_csv(metrics_path, index=False)
        print(f"\n✓ Saved all metrics to: {metrics_path}")
    
    else:
        # Load existing metrics
        metrics_path = os.path.join(args.output_dir, "all_metrics.csv")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}. Run without --skip_generation first.")
        results_df = pd.read_csv(metrics_path)
        print(f"Loaded existing metrics from: {metrics_path}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    viz_dir = os.path.join(args.output_dir, "visualizations")
    create_visualizations(results_df, viz_dir)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    summary = results_df.groupby('model').agg({
        'clip_text_similarity': ['mean', 'std'],
        'style_consistency': ['mean', 'std'],
        'image_quality': ['mean', 'std'],
        'diversity': 'mean',
    }).round(4)
    
    print("\n", summary)
    
    print("\n" + "="*80)
    print("✓ TESTING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Images: {os.path.join(args.output_dir, 'images')}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Visualizations: {viz_dir}")


if __name__ == "__main__":
    main()
