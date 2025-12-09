import torch
import numpy as np


class DPMSolverSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085,
                 beta_end: float = 0.0120):
        # 1. 初始化 Beta/Alpha
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        # 历史缓冲区：存储上一步计算的 "Data Prediction" (x0)
        self.model_outputs = []

    def set_inference_timesteps(self, num_inference_steps=20):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = []  # 重置历史

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        # 获取当前时间 t 和上一步时间 prev_t
        t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        prev_t = t - self.num_train_timesteps // self.num_inference_steps

        # 1. 获取 Alpha 和 Sigma
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        # 移至 GPU/CPU
        device = latents.device
        alpha_prod_t = alpha_prod_t.to(device)
        alpha_prod_t_prev = alpha_prod_t_prev.to(device)

        alpha_t_sqrt = alpha_prod_t ** 0.5
        sigma_t = (1 - alpha_prod_t) ** 0.5

        alpha_prev_sqrt = alpha_prod_t_prev ** 0.5
        sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

        # 2. 计算当前预测的 x0 (Data Prediction)
        # DPM-Solver++ 的核心是基于 x0 进行推导
        x0_pred = (latents - sigma_t * model_output) / alpha_t_sqrt

        # 将 x0 存入历史列表
        self.model_outputs.append(x0_pred)

        # =================================================
        # DPM-Solver++ 核心代数公式 (无 log 版本)
        # =================================================

        # 情况 1: 最后一步 (Last Step, t -> 0)
        # 此时 sigma_prev = 0。为了数值稳定，直接返回 x0
        if prev_t < 0:
            return x0_pred

        # 情况 2: 第一步 (First Step) 或历史不足
        # 降级为 DPM-Solver-1 (Euler 变换)
        if len(self.model_outputs) < 2:
            # DPM-Solver-1 公式
            # x_{t-1} = (sigma_{t-1} / sigma_t) * x_t + (alpha_{t-1} - sigma_{t-1} * alpha_t / sigma_t) * x0
            # 简化后其实就是 Euler
            prev_latents = (sigma_prev / sigma_t) * latents + (
                        alpha_prev_sqrt - (sigma_prev / sigma_t) * alpha_t_sqrt) * x0_pred

        # 情况 3: 中间步骤 (Middle Steps) -> 启用 DPM-Solver++ (2M) 二阶求解
        else:
            D0 = self.model_outputs[-1]  # 当前步 x0
            D1 = self.model_outputs[-2]  # 上一步 x0

            # 计算 r (步长比率)。假设步长均匀，r 近似为 1/2 或 1，
            # 但为了严谨，我们使用标准 DPM++ 2M 公式：
            # x_{t-1} = (sigma_{t-1}/sigma_t) * x_t + (alpha_{t-1} * (e^-h - 1)) * x0 ...
            # 为了避免 log/exp，我们直接用 Sigma/Alpha 组合代替 e^-h

            # r_k = (sigma_prev / sigma_t)
            # 这是一个简化的 2M 公式，非常稳健：

            # 第一项: 缩放当前的 latents
            term_1 = (sigma_prev / sigma_t) * latents

            # 第二项: 指向 x0 的向量 (一阶部分)
            coeff_1 = alpha_prev_sqrt - (sigma_prev / sigma_t) * alpha_t_sqrt
            term_2 = coeff_1 * D0

            # 第三项: 二阶修正 (利用 x0 的变化率)
            # 这里的 0.5 是基于线性假设的系数
            coeff_2 = coeff_1  # 在 simplified 2M 中，二阶项系数常与一阶项相关
            # 但更精确的 DPM++ 2M 修正项是：
            # 0.5 * (alpha_prev_sqrt - sigma_prev * alpha_t_sqrt / sigma_t) * (D0 - D1)
            # 这里的实现我们直接用上面的 coeff_1

            term_3 = 0.5 * coeff_1 * (D0 - D1)

            prev_latents = term_1 + term_2 - term_3

        # 维护历史长度
        if len(self.model_outputs) > 2:
            self.model_outputs.pop(0)

        return prev_latents

    def add_noise(self, original_samples, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device,
                            dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples