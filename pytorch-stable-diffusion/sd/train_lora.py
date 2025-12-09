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
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer
from torchvision import transforms
from PIL import Image
from model_loader import preload_models_with_lora, save_models_lora
from lora import get_lora_parameters

# --- 1. Dataset ---
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
        return image, caption

# --- 2. Text Encoder Helper (CLIP) ---
def encode_text(clip_model, tokenizer, captions, device):
    # Tokenize and encode captions using CLIP
    # Assumes CLIP model has encode_text method
    tokens = tokenizer(captions, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = clip_model.encode_text(tokens.input_ids)
    return text_embeds

# --- 3. Training Loop ---
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
        elif (hasattr(torch, "has_mps") and torch.has_mps) or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            device = "mps"
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")

    # Load base model and apply LoRA
    models = preload_models_with_lora(ckpt_path, device, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    # Prepare dataset and dataloader
    dataset = StyleDataset(image_dir, captions_file, tokenizer, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    lora_params = get_lora_parameters(models)
    optimizer = torch.optim.Adam(lora_params, lr=lr)

    for epoch in range(epochs):
        models["unet"].train()
        for batch in dataloader:
            images, captions = batch
            images = images.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()
            # Forward pass (customize as needed for your pipeline)
            loss = models["unet"].forward(images, captions)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} completed.")

    # Save LoRA weights in a format compatible with pipeline.generate
    save_models_lora(models, output_path)
    print(f"LoRA weights saved to {output_path}")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = StyleDataset(image_dir, captions_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load models with LoRA
    models = preload_models_with_lora(
        ckpt_path=ckpt_path,
        device=device,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        apply_to_diffusion=True,
        apply_to_clip=True
    )
    diffusion = models['diffusion']
    clip = models['clip']

    # Tokenizer for CLIP
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    # Only train LoRA parameters
    for param in diffusion.parameters():
        param.requires_grad = False
    lora_params = get_lora_parameters(diffusion)
    for param in lora_params:
        param.requires_grad = True
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # --- Training Loop ---
    for epoch in range(epochs):
        diffusion.train()
        total_loss = 0
        for images, captions in dataloader:
            images = images.to(device)
            # Encode text
            text_embeds = encode_text(clip, tokenizer, list(captions), device)
            # Forward pass (example: simple L2 loss on latent)
            # You should replace this with your actual diffusion loss
            latents = diffusion.encoder(images)
            noise = torch.randn_like(latents)
            noisy_latents = latents + noise
            pred = diffusion.decoder(noisy_latents)
            loss = torch.nn.functional.mse_loss(pred, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        # Optionally save intermediate LoRA weights
        if (epoch+1) % 5 == 0:
            save_models_lora(models, f"{output_path}_epoch{epoch+1}.pt")
    # Save final LoRA weights
    save_models_lora(models, output_path)
    print(f"Training complete. LoRA weights saved to {output_path}")

# --- 4. Main Entrypoint ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LoRA for Stable Diffusion style adaptation")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with training images')
    parser.add_argument('--captions_file', type=str, required=True, help='Captions file (img: caption per line)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to base model checkpoint')
    parser.add_argument('--output_path', type=str, default='lora_weights.pt', help='Output path for LoRA weights')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
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
        lora_dropout=args.lora_dropout
    )
