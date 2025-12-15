# run_textual_inversion_experiments.py

import os
import shutil
import random
import json
import csv
import subprocess
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from tqdm.auto import tqdm

from transformers import CLIPProcessor, CLIPModel


# ================== CONFIG ==================

DATA_ROOT = "./inputs_textual_inversion_preprocessed/style1"
SUBSETS_ROOT = "./subsets"
RESULTS_ROOT = "./subset_results"

# Experimental grid
N_SUBSETS = 3                       # how many random subsets per size
IMAGES_PER_SUBSET = [2, 3, 4, 5, 6]
TRAIN_STEPS_LIST   = [500, 1000, 1500, 2000, 2500]  # one steps value per subset size
N_TRAIN_RUNS = 2                    # repeats per subset
SEED_BASE = 1234                    # base seed to make things reproducible

PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
PLACEHOLDER_TOKEN = "<style1>"
INITIALIZER_TOKEN = "painting"

TRAIN_SCRIPT = "train_textual_inversion.py"

# Evaluation
N_EVAL_IMAGES = 4        # number of images to generate per model per prompt
EVAL_PROMPTS = [
    f"a painting in the style of {PLACEHOLDER_TOKEN}",
    f"a portrait in the style of {PLACEHOLDER_TOKEN}",
    f"a landscape in the style of {PLACEHOLDER_TOKEN}",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_all_images(data_root: Path) -> List[str]:
    return [
        f
        for f in os.listdir(data_root)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def create_subset(
    all_images: List[str],
    n_imgs: int,
    subset_idx: int,
    data_root: Path,
    subsets_root: Path,
) -> Path:
    subset_dir = subsets_root / f"subset_{n_imgs}_{subset_idx+1}"
    ensure_clean_dir(subset_dir)

    subset_images = random.sample(all_images, min(n_imgs, len(all_images)))
    for img in subset_images:
        shutil.copy(data_root / img, subset_dir / img)

    print(f"Created subset {n_imgs}_{subset_idx+1} with images: {subset_images}")

    # Save metadata
    with open(subset_dir / "subset_meta.json", "w") as f:
        json.dump({"images": subset_images}, f, indent=2)

    return subset_dir


def train_model(
    subset_dir: Path,
    train_output: Path,
    max_train_steps: int,
    train_run_seed: int,
) -> bool:
    train_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", PRETRAINED_MODEL,
        "--data_root", str(subset_dir),
        "--output_dir", str(train_output),
        "--placeholder_token", PLACEHOLDER_TOKEN,
        "--initializer_token", INITIALIZER_TOKEN,
        "--what_to_teach", "style",
        "--max_train_steps", str(max_train_steps),
        "--learning_rate", "5e-4",
        "--train_batch_size", "1",
        "--repeats", "100",
        "--image_size", "512",
        "--seed", str(train_run_seed),
    ]

    print("\n=== Training command ===")
    print(" ".join(cmd))
    print("========================\n")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed for {train_output}: {e}")
        return False


def load_clip() -> (CLIPModel, CLIPProcessor):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


def load_subset_images(subset_dir: Path, max_imgs: int = 16):
    """Load up to max_imgs training images from this subset."""
    from PIL import Image

    img_paths = [
        p for p in subset_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    img_paths = img_paths[:max_imgs]  # limit for speed

    images = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"[WARN] Could not open {p}: {e}")
    return images


@torch.no_grad()
def compute_clip_image_image_similarity(
    generated_images,
    ref_images,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> float:
    """
    Compute average CLIP image-image cosine similarity between
    generated images and reference training images (style-ish).
    """
    if len(ref_images) == 0 or len(generated_images) == 0:
        return float("nan")

    # Precompute embeddings for reference images
    ref_inputs = clip_processor(
        images=ref_images,
        return_tensors="pt",
    ).to(DEVICE)
    ref_feats = clip_model.get_image_features(**ref_inputs)
    ref_feats = ref_feats / ref_feats.norm(dim=-1, keepdim=True)

    sims = []

    for gen_img in generated_images:
        gen_inputs = clip_processor(
            images=[gen_img],
            return_tensors="pt",
        ).to(DEVICE)
        gen_feat = clip_model.get_image_features(**gen_inputs)
        gen_feat = gen_feat / gen_feat.norm(dim=-1, keepdim=True)

        sim = (gen_feat @ ref_feats.T).squeeze(0)  # [num_ref]
        sims.extend(sim.cpu().tolist())

    return float(np.mean(sims))


@torch.no_grad()
def evaluate_model(
    model_dir: Path,
    subset_dir: Path,
    prompts,
    n_images: int,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> Dict[str, float]:
    """
    Evaluation:
    - CLIP text-image similarity (prompt vs generated images)
    - CLIP image-image similarity (generated vs training subset images) as style-ish score
    - Save all generated images to model_dir / 'generated_images'
    """

    from diffusers import StableDiffusionPipeline  # import here

    pipe = StableDiffusionPipeline.from_pretrained(model_dir).to(DEVICE)
    pipe.safety_checker = None  # optionally disable to avoid filtering
    pipe.enable_attention_slicing()

    gen_dir = model_dir / "generated_images"
    gen_dir.mkdir(parents=True, exist_ok=True)

    all_generated_images = []
    text_image_scores = []

    for p_idx, prompt in enumerate(tqdm(prompts, desc=f"Evaluating {model_dir.name}")):
        images = []
        for i in range(n_images):
            out = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)
            img = out.images[0]
            images.append(img)
            all_generated_images.append(img)

            # --- Save generated image (ours, from trained model) ---
            filename = f"prompt{p_idx+1}_img{i+1}.png"
            img.save(gen_dir / filename)

        # text-image CLIP similarity
        inputs = clip_processor(
            text=[prompt] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
            dim=-1, keepdim=True
        )
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
            dim=-1, keepdim=True
        )

        sim = (image_embeds * text_embeds).sum(dim=-1)
        text_image_scores.extend(sim.cpu().tolist())

    clip_text_image_mean = float(np.mean(text_image_scores))
    clip_text_image_std = float(np.std(text_image_scores))

    # Style-ish similarity: generated vs training subset images
    ref_images = load_subset_images(subset_dir, max_imgs=16)
    clip_img_img_mean = compute_clip_image_image_similarity(
        generated_images=all_generated_images,
        ref_images=ref_images,
        clip_model=clip_model,
        clip_processor=clip_processor,
    )

    return {
        "clip_text_image_mean": clip_text_image_mean,
        "clip_text_image_std": clip_text_image_std,
        "clip_text_image_num_samples": len(text_image_scores),
        "clip_img_img_mean": clip_img_img_mean,
    }


def main():
    random.seed(SEED_BASE)

    data_root = Path(DATA_ROOT)
    subsets_root = Path(SUBSETS_ROOT)
    results_root = Path(RESULTS_ROOT)

    subsets_root.mkdir(exist_ok=True, parents=True)
    results_root.mkdir(exist_ok=True, parents=True)

    all_images = get_all_images(data_root)
    if not all_images:
        raise RuntimeError(f"No images found in {DATA_ROOT}")

    # Load CLIP once
    clip_model, clip_processor = load_clip()

    results_csv_path = results_root / "results.csv"
    csv_fieldnames = [
        "subset_size",
        "subset_index",
        "train_steps",
        "train_run",
        "seed",
        "model_dir",
        "clip_text_image_mean",
        "clip_text_image_std",
        "clip_text_image_num_samples",
        "clip_img_img_mean",  # style-ish similarity
    ]

    with open(results_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()

        # Loop over subset sizes and train steps
        for n_imgs, train_steps in zip(IMAGES_PER_SUBSET, TRAIN_STEPS_LIST):
            for subset_idx in range(N_SUBSETS):
                # make subset (reproducible seed per subset)
                subset_seed = SEED_BASE + n_imgs * 100 + subset_idx
                random.seed(subset_seed)
                subset_dir = create_subset(
                    all_images, n_imgs, subset_idx, data_root, subsets_root
                )

                for train_run in range(N_TRAIN_RUNS):
                    run_seed = SEED_BASE + n_imgs * 100 + subset_idx * 10 + train_run
                    train_output = results_root / f"subset_{n_imgs}_{subset_idx+1}_steps_{train_steps}_run_{train_run+1}"

                    success = train_model(
                        subset_dir=subset_dir,
                        train_output=train_output,
                        max_train_steps=train_steps,
                        train_run_seed=run_seed,
                    )
                    if not success:
                        writer.writerow(
                            {
                                "subset_size": n_imgs,
                                "subset_index": subset_idx + 1,
                                "train_steps": train_steps,
                                "train_run": train_run + 1,
                                "seed": run_seed,
                                "model_dir": str(train_output),
                                "clip_text_image_mean": float("nan"),
                                "clip_text_image_std": float("nan"),
                                "clip_text_image_num_samples": 0,
                                "clip_img_img_mean": float("nan"),
                            }
                        )
                        csvfile.flush()
                        continue

                    # Evaluate
                    try:
                        metrics = evaluate_model(
                            model_dir=train_output,
                            subset_dir=subset_dir,
                            prompts=EVAL_PROMPTS,
                            n_images=N_EVAL_IMAGES,
                            clip_model=clip_model,
                            clip_processor=clip_processor,
                        )
                    except Exception as e:
                        print(f"[ERROR] Evaluation failed for {train_output}: {e}")
                        metrics = {
                            "clip_text_image_mean": float("nan"),
                            "clip_text_image_std": float("nan"),
                            "clip_text_image_num_samples": 0,
                            "clip_img_img_mean": float("nan"),
                        }

                    # Save metrics JSON next to model
                    with open(train_output / "eval_metrics.json", "w") as f:
                        json.dump(metrics, f, indent=2)

                    row = {
                        "subset_size": n_imgs,
                        "subset_index": subset_idx + 1,
                        "train_steps": train_steps,
                        "train_run": train_run + 1,
                        "seed": run_seed,
                        "model_dir": str(train_output),
                        **metrics,
                    }
                    writer.writerow(row)
                    csvfile.flush()

    print(f"\nAll experiments complete. Results saved to {results_csv_path}")


if __name__ == "__main__":
    main()