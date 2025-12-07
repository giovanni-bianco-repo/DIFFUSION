# Diffusion Model Style Transfer: Training vs Inference Study

## Core Implementation
- **Base:** Stable Diffusion from scratch (PyTorch only)
- **Components:** U-Net, VAE, CLIP (load pretrained weights)
- **Custom:** LoRA module, Textual Inversion, Stochastic Inversion

## Training Experiments

### 1. LoRA Fine-tuning
Train LoRA adapters with variations in:
- **Noise schedules:** Linear, Cosine, Quadratic
- **Timestep sampling:** Uniform, SNR-weighted, High/Low-noise only
- **Loss weighting:** MSE, SNR-weighted, Min-SNR-γ
## Inference Experiments  
Test all methods with different schedulers:
- **Methods:** LoRA models, Textual Inversion tokens, Stochastic Inversion
- **Schedulers:** DDPM, DDIM, Euler, Euler-A, DPM++ 2M, DPM++ SDE
- **Variables:** Steps (10/20/50), CFG scale (3/7.5/12)
## Comparative Analysis
- **LoRA vs Textual Inversion:** Parameter efficiency vs quality tradeoff
- **Stochastic Inversion:** Content preservation vs style strength
- **Training methods × Schedulers:** Cross-method evaluation matrix

## Evaluation
- **Metrics:** CLIP content similarity, CLIP style similarity, LPIPS, runtime
- **Analysis:** Training impact vs inference impact on style transfer
- **Outputs:** Comparison grids, metric plots, statistical analysis

