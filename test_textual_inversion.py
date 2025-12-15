import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import clip

# Paths
trained_dir = "./style1-output"
standard_model = "runwayml/stable-diffusion-v1-5"

# Prompts
prompt_real = 'a dog in style of cubism'
prompt = "a dog in the style of <style1>"
base_prompt = "a dog in the style of style1"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pipelines
pipe_trained = StableDiffusionPipeline.from_pretrained(
    trained_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)

pipe_base = StableDiffusionPipeline.from_pretrained(
    standard_model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)

# Generate images
image_trained = pipe_trained(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image_base = pipe_base(base_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image_trained.save("style1_trained_output.png")
image_base.save("style1_base_output.png")
print("Saved: cubism_trained_output.png, cubism_base_output.png")

# CLIP Score comparison
model, preprocess = clip.load("ViT-B/32", device=device)

def clip_score(image: Image.Image, text: str):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        score = (image_features @ text_features.T).item()
    return score

score_trained = clip_score(image_trained, prompt_real)
score_base = clip_score(image_base, prompt_real)

print(f"CLIP Score (trained): {score_trained:.4f}")
print(f"CLIP Score (base):    {score_base:.4f}")