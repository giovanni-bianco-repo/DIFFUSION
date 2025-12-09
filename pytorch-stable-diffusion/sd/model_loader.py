from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from lora import apply_lora_to_model, load_lora_weights, save_lora_weights, get_lora_parameters

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }


def preload_models_with_lora(ckpt_path, device, lora_rank=4, lora_alpha=1.0, lora_dropout=0.0, 
                              apply_to_diffusion=True, apply_to_clip=False):
    """
    Load models and apply LoRA to specified components.
    
    Args:
        ckpt_path: Path to the base model checkpoint
        device: Device to load models on
        lora_rank: Rank for LoRA decomposition
        lora_alpha: Alpha scaling for LoRA
        lora_dropout: Dropout for LoRA layers
        apply_to_diffusion: Whether to apply LoRA to the diffusion model
        apply_to_clip: Whether to apply LoRA to the CLIP model
    
    Returns:
        Dictionary containing the models with LoRA applied
    """
    # Load base models
    models = preload_models_from_standard_weights(ckpt_path, device)
    
    # Apply LoRA to diffusion model (UNet)
    if apply_to_diffusion:
        # Target the attention layers in the diffusion model
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'in_proj']
        models['diffusion'] = apply_lora_to_model(
            models['diffusion'], 
            rank=lora_rank, 
            alpha=lora_alpha, 
            dropout=lora_dropout,
            target_modules=target_modules
        )
        print("Applied LoRA to diffusion model (UNet)")
    
    # Apply LoRA to CLIP model
    if apply_to_clip:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
        models['clip'] = apply_lora_to_model(
            models['clip'],
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules
        )
        print("Applied LoRA to CLIP model")
    
    return models


def load_lora_into_models(models, lora_path):
    """
    Load pre-trained LoRA weights into models.
    
    Args:
        models: Dictionary of models (from preload_models_with_lora)
        lora_path: Path to the LoRA checkpoint file
    """
    import torch
    lora_state_dict = torch.load(lora_path)
    
    # Load LoRA weights for each model
    for model_name in ['diffusion', 'clip', 'encoder', 'decoder']:
        if model_name in models:
            model_prefix = f"{model_name}."
            model_lora_dict = {
                k[len(model_prefix):]: v 
                for k, v in lora_state_dict.items() 
                if k.startswith(model_prefix)
            }
            if model_lora_dict:
                load_lora_weights(models[model_name], model_lora_dict)
                print(f"Loaded LoRA weights for {model_name}")
    
    return models


def save_models_lora(models, lora_path):
    """
    Save only the LoRA weights from the models.
    
    Args:
        models: Dictionary of models with LoRA
        lora_path: Path to save the LoRA checkpoint
    """
    import torch
    lora_state_dict = {}
    
    for model_name, model in models.items():
        for name, module in model.named_modules():
            from lora import LoRALayer, LoRALinear
            if isinstance(module, (LoRALayer, LoRALinear)) and hasattr(module, 'lora_A'):
                prefix = f"{model_name}.{name}"
                lora_state_dict[f"{prefix}.lora_A"] = module.lora_A.data.cpu()
                lora_state_dict[f"{prefix}.lora_B"] = module.lora_B.data.cpu()
                lora_state_dict[f"{prefix}.alpha"] = module.alpha
                lora_state_dict[f"{prefix}.rank"] = module.rank
    
    torch.save(lora_state_dict, lora_path)
    print(f"Saved LoRA weights to {lora_path} ({len(lora_state_dict)} parameters)")
    
    return lora_state_dict
