# Parameter-Efficient Style Transfer in Diffusion Models: A Comparative Study of Training and Inference Strategies

**Course:** Neural Networks and Deep Learning  
**Date:** December 2025  
**Authors:** [Your Names]

---

## 1. Introduction

Diffusion models have emerged as powerful tools for image generation and style transfer. However, full model fine-tuning is computationally expensive and time-consuming. This project investigates **Low-Rank Adaptation (LoRA)** as a parameter-efficient alternative for artistic style transfer, while systematically analyzing how both **training-time noise strategies** and **inference-time sampling methods** affect the quality and characteristics of generated images.

### Research Question

**"How do training-time noise strategies and inference-time schedulers affect style expression and content preservation in LoRA-based diffusion model fine-tuning?"**

---

## 2. Background

### 2.1 Diffusion Models

Denoising Diffusion Probabilistic Models (DDPM) learn to generate images by reversing a gradual noising process. During training, random Gaussian noise is added to images at various timesteps, and a U-Net model learns to predict and remove this noise. The noise schedule β and timestep sampling strategy significantly impact what the model learns.

### 2.2 Low-Rank Adaptation (LoRA)

LoRA achieves parameter-efficient fine-tuning by decomposing weight updates into low-rank matrices:

```
W_new = W_pretrained + B × A
```

where:
- W is the original weight matrix (e.g., 1024×1024)
- B is d×r and A is r×k (r << d,k)
- r is the rank (typically 4-16)

This reduces trainable parameters by 95-99%, enabling fast fine-tuning on consumer hardware while preserving model quality.

### 2.3 Style Transfer in Diffusion Models

Unlike GANs which require paired datasets, diffusion models can learn artistic styles from small, unpaired image collections. Cross-attention mechanisms in the U-Net allow text conditioning to guide style application, making them ideal for controlled style transfer tasks.

---

## 3. Methodology

### 3.1 Implementation Plan

We will implement the following components **from scratch**:

1. **LoRA Module** (~100 lines)
   - LoRALinear layer with low-rank decomposition
   - Automatic injection into U-Net attention layers
   - Separate save/load utilities for LoRA weights

2. **Training Pipeline** (~200 lines)
   - DDPM training loop with configurable noise schedules
   - Timestep sampling strategies
   - Loss weighting schemes

3. **Inference Pipeline** (~150 lines)
   - Multiple sampling schedulers (DDIM, Euler, DPM++)
   - Configurable CFG scale and step counts

4. **Evaluation Framework** (~100 lines)
   - CLIP content similarity
   - CLIP style similarity
   - LPIPS perceptual distance
   - Runtime measurements

### 3.2 Base Architecture

- **Model:** Stable Diffusion v1.5 (860M parameters)
- **Modified Components:** U-Net cross-attention layers only
- **Frozen Components:** VAE encoder/decoder, CLIP text encoder
- **LoRA Target Layers:** Cross-attention Q, K, V, and output projections
- **Trainable Parameters:** ~0.5-2% of total (depending on rank)

### 3.3 Dataset

**ArtBench-10** subset (Liao et al., 2022):
- ~60,000 high-quality artistic images across 10 styles
- Focus on 1-2 specific styles (e.g., Impressionism, Ukiyo-e)
- 10-50 images per style for few-shot learning experiments
- Simple captions describing artistic style and content

**Alternative:** Custom curated dataset (5-10K images) for faster iteration.

---

## 4. Experimental Design

### 4.1 Experiment 1: Training-Time Noise Strategies

**Goal:** Understand how different noise application methods affect style learning.

**Variables:**

| Variable | Options | Rationale |
|----------|---------|-----------|
| **Noise Schedule** | Linear β, Cosine β, Quadratic β | Different schedules affect signal-to-noise ratio across timesteps |
| **Timestep Sampling** | Uniform, SNR-weighted, High-noise only, Low-noise only | Controls which denoising steps the model learns best |
| **Loss Weighting** | Standard MSE, SNR-weighted, Min-SNR-γ | Balances learning across different noise levels |

**LoRA Configurations:**

| Model ID | Noise Schedule | Timestep Sampling | Loss Weighting | Rank |
|----------|---------------|-------------------|----------------|------|
| L1 | Linear | Uniform | MSE | 8 |
| L2 | Cosine | Uniform | MSE | 8 |
| L3 | Linear | SNR-weighted | MSE | 8 |
| L4 | Linear | High-noise only | MSE | 8 |
| L5 | Linear | Uniform | Min-SNR-5 | 8 |
| L6 | Linear | Uniform | MSE | 4 |
| L7 | Linear | Uniform | MSE | 16 |

**Training Parameters:**
- Learning rate: 1e-4 with cosine annealing
- Batch size: 4-8 (depending on GPU memory)
- Steps: 5,000-10,000
- Optimizer: AdamW
- Resolution: 512×512 → latent 64×64

**Expected Outcomes:**
- Cosine schedule should provide more stable training
- SNR-weighted sampling may improve fine detail preservation
- Higher ranks should capture more complex style features

### 4.2 Experiment 2: Inference-Time Scheduler Comparison

**Goal:** Evaluate how different sampling algorithms express learned styles.

**For each trained LoRA model, test:**

| Scheduler | Steps | CFG Scale | Characteristics |
|-----------|-------|-----------|-----------------|
| DDIM | 20, 50 | 7.5 | Deterministic, smooth |
| Euler | 20, 50 | 7.5 | Faster, more texture |
| Euler Ancestral | 20, 50 | 7.5 | Stochastic, varied |
| DPM++ 2M | 20, 50 | 7.5 | High quality, efficient |
| DPM++ SDE | 50 | 5.0, 7.5, 12.0 | More global coherence |

**Test Conditions:**
- Same prompt set across all experiments
- Multiple random seeds per configuration
- Both text-to-image and image-to-image tests

**Expected Outcomes:**
- Fewer steps → faster but less refined
- Higher CFG → stronger style but potential artifacts
- Stochastic samplers → more variation per seed
- Deterministic samplers → more consistent results

### 4.3 Experiment 3: Quantitative Evaluation

**Metrics:**

1. **CLIP Content Similarity**
   - Measure: Cosine similarity between CLIP embeddings of source and generated images
   - Range: 0-1 (higher = better content preservation)

2. **CLIP Style Similarity**
   - Measure: Cosine similarity between generated image and style reference
   - Range: 0-1 (higher = stronger style transfer)

3. **LPIPS Perceptual Distance**
   - Measure: Learned perceptual image patch similarity
   - Range: 0-1 (lower = more similar)

4. **Inference Time**
   - Measure: Seconds per image
   - Compare across schedulers and step counts

**Analysis:**
- Create scatter plots: content similarity vs. style similarity
- Identify Pareto-optimal configurations
- Statistical significance testing (t-tests)
- Qualitative human evaluation (optional)

---

## 5. Expected Results & Visualizations

### 5.1 Visual Outputs

1. **Training Progression Grids**
   - Show style learning over training steps
   - Compare different noise strategies side-by-side

2. **LoRA × Scheduler Comparison Matrix**
   - 7 LoRA models × 5 schedulers = 35 configurations
   - Same prompts across all cells
   - Highlight optimal combinations

3. **Parameter Sweep Visualizations**
   - CFG scale: 1.0 → 15.0 (grid of 7 images)
   - Inference steps: 10, 20, 30, 50, 100
   - LoRA rank: 4, 8, 16, 32

4. **Style Interpolation**
   - Show smooth transitions between styles
   - Demonstrate LoRA weight blending

### 5.2 Quantitative Plots

1. **Training Curves**
   - Loss vs. steps for different noise schedules
   - Learning rate schedules
   - Gradient norms

2. **Metric Comparison Radar Charts**
   - Content similarity, style similarity, LPIPS, speed
   - One chart per experimental condition

3. **Pareto Frontier**
   - X-axis: Content preservation
   - Y-axis: Style strength
   - Points colored by scheduler type

4. **Ablation Studies**
   - Effect of rank on quality
   - Effect of training steps
   - Effect of timestep sampling strategy

---

## 6. Technical Implementation

### 6.1 Hardware Requirements

- **GPU:** Single NVIDIA RTX 3090/4090 (24GB VRAM) or A100
- **RAM:** 32GB system RAM recommended
- **Storage:** 100GB for datasets, models, and outputs
- **Training Time:** 30 mins - 3 hours per LoRA model
- **Total Compute:** ~24 GPU hours for all experiments

### 6.2 Software Stack

```python
# Core dependencies
- PyTorch 2.0+
- Python 3.10+
- numpy, matplotlib, scipy

# Custom implementations
- LoRA module (from scratch)
- Training loop (from scratch)
- Noise schedulers (from scratch)
- Sampling schedulers (from scratch)

# Evaluation only
- transformers (for CLIP)
- lpips (for perceptual metrics)
```

### 6.3 Code Structure

```
pytorch-stable-diffusion/
├── sd/
│   ├── lora.py              # NEW: LoRA implementation
│   ├── train_lora.py        # NEW: Training script
│   ├── schedulers.py        # NEW: Sampling schedulers
│   ├── diffusion.py         # Modified for LoRA injection
│   ├── pipeline.py          # Modified for inference options
│   └── [existing files]
├── utils/
│   ├── metrics.py           # NEW: Evaluation metrics
│   ├── visualization.py     # NEW: Grid generation
│   └── logger.py            # NEW: Training logs
├── experiments/
│   ├── train_experiments.sh # Batch training script
│   └── eval_experiments.sh  # Batch evaluation
└── notebooks/
    ├── analysis.ipynb       # Results visualization
    └── demo.ipynb           # Interactive demos
```

---

## 7. Timeline (2 weeks)

### Week 1: Implementation

**Days 1-2:** Core LoRA Implementation
- LoRALinear module with proper initialization
- Injection utilities for U-Net
- Save/load functionality
- Unit tests

**Days 3-4:** Training Pipeline
- DDPM training loop
- Multiple noise schedules (linear, cosine, quadratic)
- Timestep sampling strategies
- Loss weighting schemes
- Basic logging and checkpointing

**Days 5-7:** Inference & Schedulers
- DDIM, Euler, Euler-A implementations
- DPM++ 2M and SDE implementations
- CFG scale control
- Batch inference utilities

### Week 2: Experiments & Analysis

**Days 8-10:** Run Experiments
- Train 7 LoRA models (Experiment 1)
- Generate images with all scheduler combinations (Experiment 2)
- Compute all metrics (Experiment 3)

**Days 11-12:** Analysis & Visualization
- Create comparison grids
- Generate plots and charts
- Statistical analysis
- Identify key findings

**Days 13-14:** Report Writing
- Results synthesis
- Discussion and conclusions
- Future work recommendations

---

## 8. Success Criteria

**Minimum Viable Project:**
✅ LoRA implementation working correctly  
✅ At least 3 trained models with different noise strategies  
✅ At least 3 different inference schedulers tested  
✅ Quantitative metrics computed  
✅ Visual comparison grids generated  

**Target Goals:**
✅ All 7 LoRA configurations trained  
✅ All 5 schedulers implemented and tested  
✅ Comprehensive metric analysis  
✅ Clear insights about training vs. inference effects  
✅ Publication-quality visualizations  

**Stretch Goals:**
⭐ Multiple artistic styles compared  
⭐ Image-to-image translation experiments  
⭐ LoRA weight merging/blending  
⭐ Interactive demo notebook  

---

## 9. Potential Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| GPU memory limitations | Use gradient checkpointing, smaller batch sizes, mixed precision (FP16) |
| Training instability | Implement gradient clipping, careful learning rate tuning, EMA weights |
| Long training times | Start with smaller subset, parallelize experiments if multiple GPUs available |
| Metric interpretation | Include qualitative human evaluation, diverse test prompts |
| Code complexity | Modular design, comprehensive documentation, incremental testing |

---

## 10. Expected Contributions

### 10.1 Technical Contributions
- Clean, documented LoRA implementation from scratch
- Multiple noise schedule implementations
- Multiple sampling scheduler implementations
- Reusable evaluation framework

### 10.2 Research Contributions
- Systematic comparison of training vs. inference factors
- Insights into parameter-efficient style transfer
- Practical guidelines for LoRA configuration
- Trade-off analysis: quality vs. speed vs. compute

### 10.3 Educational Value
- Deep understanding of diffusion model training dynamics
- Practical experience with low-rank adaptation
- Exposure to multiple sampling algorithms
- Experience with quantitative ML evaluation

---

## 11. References

[1] Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR.

[2] Ho et al. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.

[3] Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.

[4] Song et al. (2020). "Denoising Diffusion Implicit Models." ICLR.

[5] Lu et al. (2022). "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling." NeurIPS.

[6] Liao et al. (2022). "The ArtBench Dataset: Benchmarking Generative Models with Artworks." ECCV Workshops.

[7] Hang et al. (2023). "Efficient Diffusion Training via Min-SNR Weighting Strategy." arXiv.

[8] Radford et al. (2021). "Learning Transferable Visual Models from Natural Language Supervision." ICML.

[9] Zhang et al. (2023). "Inversion-Based Style Transfer with Diffusion Models." CVPR.

[10] Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models." ICML.

---

## Appendix A: Detailed LoRA Mathematics

### Forward Pass

Given input `x` and original linear layer with weights `W`:

```
# Standard linear layer
y = xW

# LoRA-enhanced layer
y = xW + x(BA) * (α/r)

where:
- W: frozen pretrained weights (d × k)
- B: trainable low-rank matrix (d × r)
- A: trainable low-rank matrix (r × k)
- α: scaling hyperparameter (default: r)
- r: rank (typically 4-16)
```

### Initialization
- `A ~ Kaiming Uniform` (similar to standard weight init)
- `B = 0` (ensures LoRA starts as identity function)
- This guarantees training starts from pretrained checkpoint behavior

### Merging for Inference
For deployment, LoRA can be merged back:
```
W_merged = W + BA * (α/r)
```
This eliminates runtime overhead while preserving learned adaptations.

---

## Appendix B: Noise Schedule Formulations

### Linear Schedule (DDPM Original)
```python
betas = torch.linspace(β_start, β_end, num_steps)
# β_start = 0.00085, β_end = 0.012
```

### Cosine Schedule (Improved DDPM)
```python
s = 0.008
t = torch.linspace(0, 1, num_steps)
alphas_cumprod = torch.cos((t + s) / (1 + s) * π/2) ** 2
betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
```

### Quadratic Schedule
```python
betas = torch.linspace(β_start**0.5, β_end**0.5, num_steps) ** 2
```

Each schedule produces different signal-to-noise ratios across timesteps, affecting what the model learns at different denoising stages.

---

**End of Proposal**
