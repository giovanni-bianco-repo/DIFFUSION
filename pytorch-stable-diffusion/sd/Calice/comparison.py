import os
import sys
import torch
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from PIL import Image
from transformers import CLIPTokenizer
import model_loader
import pipeline

DEVICE = "cpu"
ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

vocab_path = "/Users/pai/Desktop/DIFFUSION/pytorch-stable-diffusion/data/vocab.json"
merges_path = "/Users/pai/Desktop/DIFFUSION/pytorch-stable-diffusion/data/merges.txt"
model_ckpt_path = "/Users/pai/Desktop/DIFFUSION/pytorch-stable-diffusion/data/v1-5-pruned-emaonly.ckpt"

tokenizer = CLIPTokenizer(vocab_path, merges_file=merges_path)
models = model_loader.preload_models_from_standard_weights(model_ckpt_path, DEVICE)

steps_list = [100]
samplers = ["dpm","ddim","ddpm"]
fixed_seed = 42
prompt = "Hyper-realistic close-up portrait of an elderly viking warrior, deep wrinkles, weathered skin, detailed white beard, intense blue eyes, dramatic side lighting, Rembrandt lighting, 8k resolution, cinematic, raw photo, sharp focus."

output_dir = "/Users/pai/Desktop/NNDL"
os.makedirs(output_dir, exist_ok=True)

results = {
    "ddpm": {},
    "ddim": {},
    "dpm": {}
}


for sampler_name in samplers:
    for n_steps in steps_list:
        print(f"正在生成: Sampler={sampler_name}, Steps={n_steps} ...")

        output_image_array = pipeline.generate(
            prompt=prompt,
            uncond_prompt="",
            input_image=None,
            strength=0.9,
            do_cfg=True,
            cfg_scale=8,
            sampler_name=sampler_name,
            n_inference_steps=n_steps,
            seed=fixed_seed,
            models=models,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )

        img = Image.fromarray(output_image_array)

        filename = f"{sampler_name}_{n_steps}.png"
        save_path = os.path.join(output_dir, filename)
        img.save(save_path)

        results[sampler_name][n_steps] = img


rows = len(samplers)
cols = len(steps_list)
fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
fig.suptitle(f"DDPM vs DDIM Comparison (Seed={fixed_seed})", fontsize=16)

for i, sampler_name in enumerate(samplers):
    for j, n_steps in enumerate(steps_list):
        ax = axes[i, j]
        img = results[sampler_name][n_steps]

        ax.imshow(img)
        ax.set_title(f"{sampler_name.upper()} - {n_steps} Steps")
        ax.axis("off")

plt.tight_layout()
grid_path = os.path.join(output_dir, "grid_comparison.png")
plt.savefig(grid_path, dpi=150)
plt.show()
