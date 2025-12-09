import os
import torch
import matplotlib.pyplot as plt
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

vocab_path = "/data/vocab.json"
merges_path = "/data/merges.txt"
model_ckpt_path = "/data/v1-5-pruned-emaonly.ckpt"

tokenizer = CLIPTokenizer(vocab_path, merges_file=merges_path)
models = model_loader.preload_models_from_standard_weights(model_ckpt_path, DEVICE)
print("模型加载完成！")

steps_list = [2,5,7,15,20,50]
samplers = ["ddim","ddpm"]
fixed_seed = 42
prompt = "A futuristic white architectural building, Zaha Hadid style, curved glass facade, blue sky background, minimal, clean lines, sunny day, architectural photography, ultra sharp."

output_dir = "/Users/pai/Desktop/NNDL"
os.makedirs(output_dir, exist_ok=True)

results = {
    "ddpm": {},
    "ddim": {}
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
