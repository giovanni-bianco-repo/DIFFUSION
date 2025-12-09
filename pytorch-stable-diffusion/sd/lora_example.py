"""
Example script demonstrating how to use LoRA with Stable Diffusion.

This script shows:
1. Loading a model with LoRA
2. Training LoRA weights (example setup)
3. Saving and loading LoRA weights
4. Merging LoRA weights into the base model
"""

import torch
from model_loader import preload_models_with_lora, save_models_lora, load_lora_into_models
from lora import get_lora_parameters, set_lora_enabled, merge_lora_weights

def example_load_model_with_lora():
    """Example: Load a stable diffusion model with LoRA applied"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models with LoRA
    # This applies LoRA to the attention layers in the UNet
    models = preload_models_with_lora(
        ckpt_path="data/v1-5-pruned-emaonly.ckpt",
        device=device,
        lora_rank=4,           # Rank of LoRA decomposition (lower = fewer parameters)
        lora_alpha=1.0,        # Scaling factor
        lora_dropout=0.0,      # Dropout for regularization
        apply_to_diffusion=True,  # Apply to UNet
        apply_to_clip=False    # Don't apply to CLIP (usually not needed)
    )
    
    return models


def example_get_lora_parameters(models):
    """Example: Get only the LoRA parameters for training"""
    
    # Get LoRA parameters from the diffusion model
    lora_params = get_lora_parameters(models['diffusion'])
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in lora_params)
    print(f"Number of trainable LoRA parameters: {total_params:,}")
    
    # You can now use these parameters with an optimizer
    # optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    
    return lora_params


def example_training_setup(models):
    """Example: Set up for LoRA training"""
    
    # Freeze all base model parameters
    for param in models['diffusion'].parameters():
        param.requires_grad = False
    
    # Unfreeze only LoRA parameters
    lora_params = get_lora_parameters(models['diffusion'])
    for param in lora_params:
        param.requires_grad = True
    
    # Set up optimizer to train only LoRA parameters
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in lora_params)
    total_params = sum(p.numel() for p in models['diffusion'].parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    return optimizer


def example_save_lora_weights(models, save_path="lora_weights.pt"):
    """Example: Save only the LoRA weights (not the full model)"""
    
    # This saves only the LoRA matrices, creating a very small checkpoint
    save_models_lora(models, save_path)
    
    # Check file size
    import os
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"LoRA checkpoint size: {file_size_mb:.2f} MB")


def example_load_lora_weights(models, lora_path="lora_weights.pt"):
    """Example: Load pre-trained LoRA weights"""
    
    # Load LoRA weights into models
    models = load_lora_into_models(models, lora_path)
    print("Successfully loaded LoRA weights")
    
    return models


def example_toggle_lora(models):
    """Example: Enable/disable LoRA at inference time"""
    
    # Disable LoRA (use only base model)
    set_lora_enabled(models['diffusion'], enabled=False)
    print("LoRA disabled - using base model only")
    
    # Re-enable LoRA
    set_lora_enabled(models['diffusion'], enabled=True)
    print("LoRA enabled - using adapted model")


def example_merge_lora(models):
    """Example: Merge LoRA weights into base model"""
    
    # This permanently adds the LoRA adaptations to the base weights
    # After merging, you get a single model without LoRA layers
    merge_lora_weights(models['diffusion'])
    print("LoRA weights merged into base model")
    
    # The model now behaves the same but without LoRA overhead
    # You can save this as a new checkpoint if desired


def example_inference_with_lora(models):
    """Example: Generate images using the LoRA-adapted model"""
    
    # Import pipeline
    from pipeline import generate
    
    # Generate with LoRA
    set_lora_enabled(models['diffusion'], enabled=True)
    
    # Your generation code here
    # output = generate(
    #     prompt="a beautiful landscape",
    #     uncond_prompt="",
    #     models=models,
    #     ...
    # )
    
    print("Generate images with LoRA-adapted model")


def example_compare_base_vs_lora(models):
    """Example: Compare outputs with and without LoRA"""
    
    from pipeline import generate
    
    prompt = "a beautiful mountain landscape at sunset"
    seed = 42
    
    # Generate without LoRA (base model)
    set_lora_enabled(models['diffusion'], enabled=False)
    print("Generating with base model...")
    # base_output = generate(prompt=prompt, seed=seed, models=models, ...)
    
    # Generate with LoRA
    set_lora_enabled(models['diffusion'], enabled=True)
    print("Generating with LoRA-adapted model...")
    # lora_output = generate(prompt=prompt, seed=seed, models=models, ...)
    
    print("Compare the two outputs to see the effect of LoRA")


if __name__ == "__main__":
    print("=" * 60)
    print("LoRA for Stable Diffusion - Example Usage")
    print("=" * 60)
    
    # 1. Load model with LoRA
    print("\n1. Loading model with LoRA...")
    models = example_load_model_with_lora()
    
    # 2. Get LoRA parameters
    print("\n2. Getting LoRA parameters...")
    lora_params = example_get_lora_parameters(models)
    
    # 3. Set up for training
    print("\n3. Setting up for LoRA training...")
    optimizer = example_training_setup(models)
    
    # 4. Save LoRA weights
    print("\n4. Saving LoRA weights...")
    example_save_lora_weights(models, "example_lora.pt")
    
    # 5. Toggle LoRA on/off
    print("\n5. Toggling LoRA...")
    example_toggle_lora(models)
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    
    print("\nKey Benefits of LoRA:")
    print("- Train with much fewer parameters (typically 0.1-1% of full model)")
    print("- Much faster training and lower memory requirements")
    print("- LoRA weights are very small (few MB instead of several GB)")
    print("- Can easily share and swap different LoRA adaptations")
    print("- Can merge LoRA into base model for deployment")
