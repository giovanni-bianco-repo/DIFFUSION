"""
LoRA Training Script for Stable Diffusion

- Loads images and captions
- Prepares data for training
- Sets up model with LoRA
- Trains only LoRA parameters
- Saves LoRA weights compatible with pipeline.generate
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer
from torchvision import transforms
from PIL import Image

from model_loader import preload_models_with_lora, save_models_lora
from lora import get_lora_parameters
from pipeline import get_time_embedding


# --- 1. Dataset (unchanged) ---
class StyleDataset(Dataset):
    def __init__(self, image_dir, captions_file, image_size=512):
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # Read captions
        with open(captions_file, 'r') as f:
            lines = f.readlines()
        self.samples = []
        for line in lines:
            if ':' in line:
                img, cap = line.strip().split(':', 1)
                self.samples.append((img.strip(), cap.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return {'image': image, 'caption': caption}


# (kept for compatibility, but not used in the new loop)
def encode_text(clip_model, tokenizer, captions, device):
    tokens = tokenizer(captions, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = clip_model(tokens.input_ids)
    return text_embeds


# --- 2. Simple DDPM Noise Scheduler ---
class SimpleNoiseScheduler:
    """
    Minimal beta schedule / alpha_bar for forward diffusion.
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # [T]

    def add_noise(self, x0, noise, timesteps):
        """
        x0: clean latents  [B, 4, H/8, W/8]
        noise: eps ~ N(0, I) with same shape
        timesteps: LongTensor [B]
        """
        alphas_cumprod_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return torch.sqrt(alphas_cumprod_t) * x0 + torch.sqrt(1 - alphas_cumprod_t) * noise

def train_lora(
    image_dir,
    captions_file,
    ckpt_path,
    output_path,
    batch_size=4,
    epochs=10,
    lr=1e-4,
    lora_rank=4,
    lora_alpha=1.0,
    lora_dropout=0.0,
    device=None
):
    # Device selection logic
    if device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif (hasattr(torch, "has_mps") and torch.has_mps) or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            device = "mps"
    print(f"Using device: {device}")

    tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")

    models = preload_models_with_lora(
        ckpt_path,
        device,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    for key, model in models.items():
        model.to(device)
    
    clip = models["clip"]
    encoder = models["encoder"]
    diffusion = models["diffusion"]

    dataset = StyleDataset(image_dir, captions_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for p in diffusion.parameters():
        p.requires_grad = False
    lora_params = get_lora_parameters(diffusion)
    for p in lora_params:
        p.requires_grad = True

    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    noise_scheduler = SimpleNoiseScheduler(num_timesteps=1000, device=device)

    diffusion.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)        # [B, 3, H, W]
            captions = list(batch["caption"])         # list of strings
            B, _, H, W = images.shape

            # --- 1. Tokenize + encode text with CLIP (context for cross-attention) ---
            tokens = tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            with torch.no_grad():
                context = clip(tokens)   # (B, 77, 768)

            # --- 2. Encode images into latent space with VAE Encoder ---
            # Encoder expects a noise tensor and internally does mean/logvar sampling + 0.18215 scaling
            #   x = mean + stdev * noise; x *= 0.18215  :contentReference[oaicite:4]{index=4}
            with torch.no_grad():
                noise_for_encoder = torch.randn(
                    B, 4, H // 8, W // 8, device=device
                )
                latents = encoder(images, noise_for_encoder)  # [B, 4, H/8, W/8]

            # --- 3. Sample timesteps and add noise (DDPM forward process) ---
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (B,), device=device, dtype=torch.long
            )
            # A single time embedding for the batch (shape (1, 320)) to match Diffusion.forward comment :contentReference[oaicite:5]{index=5}
            t_for_embed = timesteps[:1]
            time_embedding = get_time_embedding(t_for_embed).to(device)

            # Sample epsilon and construct noisy latents
            eps = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, eps, timesteps)

            # --- 4. Predict noise with Diffusion UNet ---
            optimizer.zero_grad()
            eps_pred = diffusion(noisy_latents, context, time_embedding)  # [B, 4, H/8, W/8]

            # --- 5. Noise prediction loss ---
            loss = F.mse_loss(eps_pred, eps)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()

            batch_size_actual = images.size(0)
            epoch_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(num_samples, 1)
        print(f"Epoch {epoch+1}/{epochs} - Avg loss: {avg_loss:.4f}")

    # Save only LoRA weights (models dict passed to your helper)
    save_models_lora(models, output_path)
    print(f"Training complete. LoRA weights saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train LoRA for Stable Diffusion style adaptation"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory with training images"
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        required=True,
        help="Captions file (img: caption per line)",
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to base model checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="lora_weights.pt",
        help="Output path for LoRA weights",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    args = parser.parse_args()

    train_lora(
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        ckpt_path=args.ckpt_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
