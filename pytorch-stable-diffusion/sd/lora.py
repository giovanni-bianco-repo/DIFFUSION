import torch
from torch import nn
import math

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer that can be applied to any linear layer.
    
    Args:
        original_layer: The original nn.Linear layer to adapt
        rank: The rank of the low-rank decomposition (default: 4)
        alpha: LoRA scaling factor (default: 1.0)
        dropout: Dropout probability for LoRA layers (default: 0.0)
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Get dimensions from the original layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # LoRA decomposition: W = W_0 + (alpha/r) * B * A
        # A: (in_features, rank)
        # B: (rank, out_features)
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize A with kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Flag to enable/disable LoRA
        self.lora_enabled = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original layer output
        result = self.original_layer(x)
        
        # Add LoRA adaptation if enabled
        if self.lora_enabled and self.rank > 0:
            # x @ A @ B with scaling
            lora_out = (self.dropout(x) @ self.lora_A) @ self.lora_B
            result = result + lora_out * self.scaling
            
        return result
    
    def enable_lora(self):
        """Enable LoRA adaptation"""
        self.lora_enabled = True
        
    def disable_lora(self):
        """Disable LoRA adaptation (use only original weights)"""
        self.lora_enabled = False
        
    def merge_weights(self):
        """Merge LoRA weights into the original layer (destructive operation)"""
        if self.rank > 0 and self.lora_enabled:
            # Compute the low-rank update: (alpha/r) * B * A
            lora_weight = (self.lora_B.T @ self.lora_A.T) * self.scaling
            # Add to original weights
            self.original_layer.weight.data += lora_weight
            # Zero out LoRA matrices after merging
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)
            
    def get_lora_state_dict(self):
        """Get only the LoRA parameters"""
        return {
            'lora_A': self.lora_A,
            'lora_B': self.lora_B,
            'alpha': self.alpha,
            'rank': self.rank,
        }


class LoRALinear(nn.Module):
    """
    A linear layer with integrated LoRA support.
    Can be used as a drop-in replacement for nn.Linear when you want LoRA adaptation.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.rank = rank
        
        if rank > 0:
            self.alpha = alpha
            self.scaling = alpha / rank
            self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            
            self.lora_enabled = True
        else:
            self.lora_enabled = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        
        if self.rank > 0 and self.lora_enabled:
            lora_out = (self.dropout(x) @ self.lora_A) @ self.lora_B
            result = result + lora_out * self.scaling
            
        return result
    
    def enable_lora(self):
        self.lora_enabled = True
        
    def disable_lora(self):
        self.lora_enabled = False


def apply_lora_to_model(model: nn.Module, rank: int = 4, alpha: float = 1.0, 
                        dropout: float = 0.0, target_modules=None):
    """
    Apply LoRA to all linear layers in a model (or specific target modules).
    
    Args:
        model: The model to apply LoRA to
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout probability
        target_modules: List of module name patterns to apply LoRA to (e.g., ['q_proj', 'v_proj', 'k_proj'])
                       If None, applies to all Linear layers
    
    Returns:
        The model with LoRA applied
    """
    if target_modules is None:
        # Apply to all linear layers
        target_modules = []
    
    lora_layers = []
    
    for name, module in model.named_modules():
        # Check if this module should get LoRA
        should_apply = False
        if len(target_modules) == 0:
            # Apply to all Linear layers
            should_apply = isinstance(module, nn.Linear)
        else:
            # Apply only to specified modules
            should_apply = any(target in name for target in target_modules) and isinstance(module, nn.Linear)
        
        if should_apply:
            # Get parent module and attribute name
            *parent_names, attr_name = name.split('.')
            parent = model
            for parent_name in parent_names:
                parent = getattr(parent, parent_name)
            
            # Replace with LoRA layer
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_layer)
            lora_layers.append((name, lora_layer))
    
    print(f"Applied LoRA to {len(lora_layers)} layers")
    for name, _ in lora_layers:
        print(f"  - {name}")
    
    return model


def get_lora_parameters(model: nn.Module):
    """
    Get all LoRA parameters from a model.
    Useful for training only the LoRA weights while keeping the base model frozen.
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, (LoRALayer, LoRALinear)):
            if hasattr(module, 'lora_A'):
                lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """
    Save LoRA weights, keeping the 'unet.' prefix but removing 'diffusion.'.
    This makes the checkpoint compatible between training and inference.
    """
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, (LoRALayer, LoRALinear)) and hasattr(module, "lora_A"):

            clean_name = name

            # Remove ONLY the "diffusion." prefix
            if clean_name.startswith("diffusion."):
                clean_name = clean_name[len("diffusion."):]

            # KEEP "unet." prefix
            # So "diffusion.unet.xxx" becomes "unet.xxx"
            # And "unet.xxx" remains "unet.xxx"

            lora_state_dict[f"{clean_name}.lora_A"] = module.lora_A.data.cpu()
            lora_state_dict[f"{clean_name}.lora_B"] = module.lora_B.data.cpu()
            lora_state_dict[f"{clean_name}.alpha"] = module.alpha
            lora_state_dict[f"{clean_name}.rank"] = module.rank

    torch.save(lora_state_dict, path)
    print(f"[LoRA Saved] -> {path}, entries: {len(lora_state_dict)}")


def load_lora_weights(model: nn.Module, path: str):
    """
    Load LoRA weights into a model.
    """
    lora_state_dict = torch.load(path)
    for name, module in model['diffusion'].named_modules():
        if isinstance(module, (LoRALayer, LoRALinear)):
            lora_a_key = f"diffusion.{name}.lora_A"
            lora_b_key = f"diffusion.{name}.lora_B"
            if lora_a_key in lora_state_dict:
                module.lora_A.data = lora_state_dict[lora_a_key]
                module.lora_B.data = lora_state_dict[lora_b_key]
                if f"diffusion.{name}.alpha" in lora_state_dict:
                    module.alpha = lora_state_dict[f"diffusion.{name}.alpha"]
                    print("alpha worked")
                if f"diffusion.{name}.rank" in lora_state_dict:
                    module.rank = lora_state_dict[f"diffusion.{name}.rank"]
                    print("rank worked")
                    module.scaling = module.alpha / module.rank
    
    print(f"Loaded LoRA weights from {path}")
    return model

def merge_lora_weights(model: nn.Module):
    """
    Merge all LoRA weights into the base model weights.
    After this operation, the model will behave the same but LoRA will be disabled.
    """
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.merge_weights()
        elif isinstance(module, LoRALinear) and module.rank > 0:
            lora_weight = (module.lora_B.T @ module.lora_A.T) * module.scaling
            module.linear.weight.data += lora_weight
            nn.init.zeros_(module.lora_A)
            nn.init.zeros_(module.lora_B)
    
    print("Merged all LoRA weights into base model")


def set_lora_enabled(model: nn.Module, enabled: bool):
    """
    Enable or disable LoRA for all LoRA layers in the model.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (LoRALayer, LoRALinear)):
            if enabled:
                module.enable_lora()
            else:
                module.disable_lora()
            count += 1
    
    status = "enabled" if enabled else "disabled"
    print(f"LoRA {status} for {count} layers")
