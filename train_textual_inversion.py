# Textual Inversion Training Script for Style Embedding
# This script is based on the DigitalOcean tutorial and adapted for style training.

import os
import math
import itertools
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, DDPMScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from accelerate import Accelerator

# --- Settings ---
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"  # or local path
save_path = "./inputs_textual_inversion_preprocessed/style1"
placeholder_token = "<style1>"
initializer_token = "painting"  # a word that represents your style
what_to_teach = "style"  # "object" or "style"
output_dir = "./style1-output"

# --- Prompt templates (style) ---
imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

# --- Dataset class ---
class TextualInversionDataset(Dataset):
    def __init__(self, data_root, tokenizer, learnable_property="style", size=512, repeats=100, interpolation="bicubic", flip_p=0.5, set="train", placeholder_token="*", center_crop=False):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats if set == "train" else self.num_images
        self.interpolation = {
            "linear": Image.BILINEAR,  # LINEAR is deprecated, use BILINEAR
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }[interpolation]
        self.templates = imagenet_style_templates_small if learnable_property == "style" else []
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
    def __len__(self):
        return self._length
    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        img = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)
        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

# --- Training function ---
def training_function():
    # Load tokenizer and add placeholder token
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {placeholder_token}. Please use a different placeholder_token.")
    # Get token ids
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    # Load models
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    # Freeze all except new embedding
    def freeze_params(params):
        for param in params:
            param.requires_grad = False
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)
    # Dataset and dataloader
    train_dataset = TextualInversionDataset(
        data_root=save_path,
        tokenizer=tokenizer,
        size=512,
        placeholder_token=placeholder_token,
        repeats=100,
        learnable_property=what_to_teach,
        center_crop=False,
        set="train",
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # Scheduler and optimizer
    noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    learning_rate = 5e-4
    max_train_steps = 1000
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=learning_rate)
    text_encoder, optimizer, train_dataloader = accelerator.prepare(text_encoder, optimizer, train_dataloader)
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    vae.eval()
    unet.eval()
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    for step, batch in enumerate(train_dataloader):
        if global_step >= max_train_steps:
            break
        text_encoder.train()
        with accelerator.accumulate(text_encoder):
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
            latents = latents * 0.18215
            noise = torch.randn(latents.shape).to(latents.device)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            accelerator.backward(loss)
            if accelerator.num_processes > 1:
                grads = text_encoder.module.get_input_embeddings().weight.grad
            else:
                grads = text_encoder.get_input_embeddings().weight.grad
            index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
            grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)
            optimizer.step()
            optimizer.zero_grad()
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
        if global_step >= max_train_steps:
            break
    accelerator.wait_for_everyone()
    # Save pipeline and embedding
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline(
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        pipeline.save_pretrained(output_dir)
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
        learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, os.path.join(output_dir, "learned_embeds.bin"))
    print(f"Training complete. Embedding and pipeline saved to {output_dir}")

if __name__ == "__main__":
    training_function()
