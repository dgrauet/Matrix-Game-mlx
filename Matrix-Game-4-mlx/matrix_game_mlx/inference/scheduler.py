"""Flow-matching scheduler (MLX port of matrix_game/inference/scheduler.py, inference subset)."""
import mlx.core as mx
import numpy as np


def torch_linspace_f32(start: float, end: float, steps: int) -> np.ndarray:
    """Bitwise torch.linspace (float32, CPU kernel): float32 step, then fma(step, i, start) for the first half and
    fma(-step, steps - 1 - i, end) for the second. The float64 product and sum are exact here (24-bit step times
    an index < 2^29), so one rounding to float32 equals the fused float32 FMA.
    """
    start, end = np.float32(start), np.float32(end)
    if steps == 1:
        return np.array([start], dtype=np.float32)
    step = np.float32((end - start) / np.float32(steps - 1))
    i = np.arange(steps, dtype=np.float64)
    first = np.float64(start) + np.float64(step) * i
    second = np.float64(end) - np.float64(step) * (steps - 1 - i)
    return np.where(i < steps // 2, first, second).astype(np.float32)


class FlowMatchScheduler:
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0,
                 sigma_min=0.003 / 1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        """Build the shifted sigma grid and the matching timesteps (float32, bitwise as upstream torch)."""
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        n = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
        sigmas = torch_linspace_f32(sigma_start, self.sigma_min, n)
        if self.extra_one_step:
            sigmas = sigmas[:-1]
        if self.inverse_timesteps:
            sigmas = sigmas[::-1]
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if self.reverse_sigmas:
            sigmas = 1 - sigmas
        self.sigmas = mx.array(sigmas.astype(np.float32))
        self.timesteps = self.sigmas * self.num_train_timesteps

    def add_noise(self, original_samples: mx.array, noise: mx.array, timestep: mx.array) -> mx.array:
        """Diffusion forward corruption: (1 - sigma) * x0 + sigma * noise, sigma looked up from `timestep`."""
        if timestep.ndim == 2:
            timestep = timestep.reshape(-1)
        timestep_id = mx.argmin(mx.abs(self.timesteps[None, :] - timestep[:, None]), axis=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.astype(noise.dtype)
