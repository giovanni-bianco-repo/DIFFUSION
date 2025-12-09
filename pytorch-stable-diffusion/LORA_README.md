# LoRA Implementation for Stable Diffusion

This implementation adds **LoRA (Low-Rank Adaptation)** support to the Stable Diffusion model, enabling efficient fine-tuning with minimal parameters.

## What is LoRA?

LoRA is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into each layer. Instead of fine-tuning all parameters, LoRA trains only a small number of additional parameters.

### Key Advantages

- **Efficiency**: Train only 0.1-1% of the parameters compared to full fine-tuning
- **Speed**: Faster training and lower memory requirements
- **Small Checkpoints**: LoRA weights are typically just a few MB instead of several GB
- **Modularity**: Easy to swap different LoRA adaptations on the same base model
- **Flexibility**: Can merge LoRA weights into the base model or use them separately

## Files Added

- **`lora.py`**: Core LoRA implementation with `LoRALayer` and utility functions
- **`lora_example.py`**: Example script showing how to use LoRA
- **`LORA_README.md`**: This documentation file

## Files Modified

- **`attention.py`**: Updated `SelfAttention` and `CrossAttention` to support LoRA
- **`model_loader.py`**: Added functions to load models with LoRA and manage LoRA weights

## Quick Start

### 1. Load Model with LoRA

```python
from model_loader import preload_models_with_lora

# Load stable diffusion with LoRA applied to attention layers
models = preload_models_with_lora(
    ckpt_path="data/v1-5-pruned-emaonly.ckpt",
    device="cuda",
    lora_rank=4,           # Rank of decomposition (4, 8, 16, 32, etc.)
    lora_alpha=1.0,        # Scaling factor
    lora_dropout=0.0,      # Dropout for regularization
    apply_to_diffusion=True,  # Apply to UNet
    apply_to_clip=False    # Usually don't apply to CLIP
)
```

### 2. Set Up for Training

```python
from lora import get_lora_parameters

# Freeze base model
for param in models['diffusion'].parameters():
    param.requires_grad = False

# Get LoRA parameters
lora_params = get_lora_parameters(models['diffusion'])

# Unfreeze LoRA parameters
for param in lora_params:
    param.requires_grad = True

# Create optimizer for LoRA parameters only
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
```

### 3. Training Loop

```python
# Your training loop
for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass (only LoRA parameters will be updated)
    loss = your_loss_function(batch, models)
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

### 4. Save LoRA Weights

```python
from model_loader import save_models_lora

# Save only LoRA weights (small file ~3-10 MB)
save_models_lora(models, "my_lora_weights.pt")
```

### 5. Load LoRA Weights

```python
from model_loader import load_lora_into_models

# Load pre-trained LoRA weights
models = load_lora_into_models(models, "my_lora_weights.pt")
```

### 6. Generate with LoRA

```python
from lora import set_lora_enabled
from pipeline import generate

# Enable LoRA
set_lora_enabled(models['diffusion'], enabled=True)

# Generate images
output = generate(
    prompt="a beautiful landscape",
    models=models,
    # ... other parameters
)
```

## Advanced Usage

### Compare Base Model vs LoRA

```python
from lora import set_lora_enabled

# Generate without LoRA (base model)
set_lora_enabled(models['diffusion'], enabled=False)
base_output = generate(prompt="...", models=models)

# Generate with LoRA
set_lora_enabled(models['diffusion'], enabled=True)
lora_output = generate(prompt="...", models=models)
```

### Merge LoRA into Base Model

```python
from lora import merge_lora_weights

# Permanently merge LoRA adaptations into base weights
merge_lora_weights(models['diffusion'])

# Now the model behaves the same but without LoRA layers
# Good for deployment when you don't need to toggle LoRA
```

### Apply LoRA to Specific Layers

```python
from lora import apply_lora_to_model

# Apply LoRA only to specific layers (e.g., only Q and V projections)
model = apply_lora_to_model(
    model,
    rank=8,
    alpha=1.0,
    target_modules=['q_proj', 'v_proj']  # Only these layers
)
```

### Custom LoRA Configuration

```python
from lora import LoRALinear

# Create a custom linear layer with LoRA
layer = LoRALinear(
    in_features=768,
    out_features=768,
    bias=True,
    rank=8,           # Higher rank = more parameters, more capacity
    alpha=16.0,       # Higher alpha = stronger LoRA effect
    dropout=0.1       # Dropout for regularization
)
```

## LoRA Hyperparameters

### Rank (`r`)
- **Range**: 1-128 (typically 4, 8, 16, 32, 64)
- **Lower rank**: Fewer parameters, faster training, less capacity
- **Higher rank**: More parameters, more capacity, closer to full fine-tuning
- **Recommended**: Start with 4-8, increase if needed

### Alpha (`α`)
- **Range**: 1-64 (typically equal to or 2x the rank)
- **Controls**: The scaling of LoRA updates
- **Formula**: LoRA contribution is scaled by `α/r`
- **Recommended**: Set equal to rank, or 2x rank for stronger effects

### Dropout
- **Range**: 0.0-0.3
- **Purpose**: Regularization to prevent overfitting
- **Recommended**: 0.0 for small datasets, 0.05-0.1 for larger datasets

## Use Cases

### Style Transfer
Fine-tune on a specific art style with minimal data:
```python
# Train LoRA on a specific artistic style
# Rank 4-8 is often sufficient
models = preload_models_with_lora(ckpt_path, device, lora_rank=4)
```

### Concept Learning
Teach the model new concepts or characters:
```python
# Higher rank for learning complex concepts
models = preload_models_with_lora(ckpt_path, device, lora_rank=16)
```

### Domain Adaptation
Adapt the model to a specific domain (e.g., medical images, architecture):
```python
# Medium rank for domain adaptation
models = preload_models_with_lora(ckpt_path, device, lora_rank=8, lora_alpha=16)
```

## Memory and Performance

### Parameter Comparison

For a typical Stable Diffusion v1.5 UNet:
- **Full model**: ~860M parameters
- **LoRA (rank 4)**: ~1-2M parameters (~0.2%)
- **LoRA (rank 8)**: ~2-4M parameters (~0.4%)
- **LoRA (rank 16)**: ~4-8M parameters (~0.8%)

### File Size Comparison

- **Full checkpoint**: 4-7 GB
- **LoRA (rank 4)**: 3-5 MB
- **LoRA (rank 8)**: 6-10 MB
- **LoRA (rank 16)**: 12-20 MB

### Memory Usage

Training with LoRA requires significantly less GPU memory since:
1. Base model weights are frozen (no gradient storage)
2. Only LoRA parameters have gradients
3. Smaller optimizer state

Typical GPU memory savings: 40-60% compared to full fine-tuning

## Tips and Best Practices

1. **Start Small**: Begin with rank 4 and increase if results aren't satisfactory
2. **Match Alpha to Rank**: Setting `alpha = rank` is a good default
3. **Monitor Overfitting**: Use dropout if training on small datasets
4. **Save Frequently**: LoRA checkpoints are small, save often
5. **Experiment**: Try different rank/alpha combinations for your use case
6. **Layer Selection**: For SD, focus on attention layers (q_proj, k_proj, v_proj, out_proj)
7. **Merge for Deployment**: Merge LoRA weights into base model for inference-only deployments
8. **Multiple LoRAs**: You can load and blend multiple LoRA adaptations

## Troubleshooting

### LoRA has no effect
- Increase `lora_alpha` (try 2x or 4x the rank)
- Increase learning rate
- Check that LoRA is enabled: `set_lora_enabled(model, True)`

### Overfitting
- Decrease `lora_rank`
- Add `lora_dropout` (0.05-0.1)
- Reduce training steps

### Out of memory
- Decrease `lora_rank`
- Use gradient checkpointing
- Reduce batch size

### Training is slow
- LoRA should be faster than full fine-tuning
- Check that base model is frozen
- Verify only LoRA parameters are being updated

## References

- **Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Original Implementation**: https://github.com/microsoft/LoRA
- **Diffusers LoRA**: https://huggingface.co/docs/diffusers/training/lora

## Examples

See `lora_example.py` for comprehensive usage examples including:
- Loading models with LoRA
- Training setup
- Saving/loading LoRA weights
- Toggling LoRA on/off
- Merging weights
- Inference examples
