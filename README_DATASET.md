# Documentation: Creating a Style Dataset for Textual Inversion

## Steps performed

1. **Created dataset directory**: `inputs_textual_inversion`.
2. **Downloaded sample style images**: Used `download_style_images.py` to fetch 3 example images from Unsplash. You can replace these URLs with your own style images.
3. **Prepared prompt templates**: See `prompt_templates.py` for a list of style prompt templates suitable for textual inversion training.

## How to use your own images
- Replace the URLs in `download_style_images.py` with links to your own style images.
- Run the script again: `python3 download_style_images.py`


## Training Instructions for Textual Inversion (Style)

1. **Install dependencies** (if not already):
	```bash
	pip install diffusers transformers accelerate torch torchvision tqdm
	```
2. **Download the Stable Diffusion v1-5 model** from HuggingFace (requires a HuggingFace account and acceptance of the license):
	- Visit https://huggingface.co/runwayml/stable-diffusion-v1-5 and follow instructions to log in and get your token.
	- The script will download the model automatically if you have access.
3. **Edit `train_textual_inversion.py`** if you want to change the placeholder token, initializer token, or dataset directory.
4. **Run the training script:**
	```bash
	python3 train_textual_inversion.py
	```
5. **After training:**
	- The learned embedding and pipeline will be saved in `style-concept-output/`.
	- You can use the learned embedding in your prompts as `<my-style>`.

**Tip:** For best results, use a GPU. Training on CPU will be very slow.
