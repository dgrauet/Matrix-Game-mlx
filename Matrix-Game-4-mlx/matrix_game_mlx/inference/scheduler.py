"""Flow-matching scheduler (MLX port of matrix_game/inference/scheduler.py, inference subset)."""
import mlx.core as mx
import numpy as np


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
        """Build the shifted sigma grid and the matching timesteps (float32)."""
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        n = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
        sigmas = np.linspace(sigma_start, self.sigma_min, n, dtype=np.float32)
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
