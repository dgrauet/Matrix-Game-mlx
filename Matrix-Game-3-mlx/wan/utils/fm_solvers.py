# Ported from Matrix-Game-3/wan/utils/fm_solvers.py
# Flow matching DPM-Solver++ scheduler — MLX port (no diffusers dependency)

import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from collections import namedtuple

SchedulerOutput = namedtuple("SchedulerOutput", ["prev_sample"])


def get_sampling_sigmas(sampling_steps: int, shift: float) -> np.ndarray:
    """Compute the sigma schedule for flow matching sampling.

    Args:
        sampling_steps: Number of sampling steps.
        shift: Shift factor for the sigma schedule.

    Returns:
        Numpy array of sigma values.
    """
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    timesteps: Optional[List[float]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple:
    """Retrieve timesteps from a scheduler.

    Args:
        scheduler: The scheduler instance.
        num_inference_steps: Number of inference steps.
        timesteps: Custom timestep schedule.
        sigmas: Custom sigma schedule.

    Returns:
        Tuple of (timesteps, num_inference_steps).
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed."
        )
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FlowDPMSolverMultistepScheduler:
    """Flow matching DPM-Solver++ multistep scheduler.

    A fast dedicated high-order solver for diffusion ODEs adapted for
    flow matching. Supports dpmsolver++ and sde-dpmsolver++ algorithms
    with midpoint or heun solver types.

    Args:
        num_train_timesteps: Number of diffusion steps to train the model.
        solver_order: DPMSolver order (1, 2, or 3).
        prediction_type: Must be ``"flow_prediction"``.
        shift: Factor to adjust sigmas in the noise schedule.
        use_dynamic_shifting: Whether to apply dynamic shifting based on resolution.
        thresholding: Whether to use dynamic thresholding.
        dynamic_thresholding_ratio: Ratio for dynamic thresholding.
        sample_max_value: Threshold value for dynamic thresholding.
        algorithm_type: Algorithm type (``"dpmsolver++"`` or ``"sde-dpmsolver++"``).
        solver_type: Solver type (``"midpoint"`` or ``"heun"``).
        lower_order_final: Whether to use lower-order solvers in final steps.
        euler_at_final: Whether to use Euler's method in the final step.
        final_sigmas_type: Final sigma value type (``"zero"`` or ``"sigma_min"``).
        lambda_min_clipped: Clipping threshold for lambda minimum.
        variance_type: Set to ``"learned"`` or ``"learned_range"`` if model predicts variance.
        invert_sigmas: Whether to invert the sigma schedule.
    """

    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        use_dynamic_shifting: bool = False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        final_sigmas_type: Optional[str] = "zero",
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        invert_sigmas: bool = False,
    ):
        # Validate and fix algorithm_type
        if algorithm_type not in [
            "dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"
        ]:
            if algorithm_type == "deis":
                algorithm_type = "dpmsolver++"
            else:
                raise NotImplementedError(
                    f"{algorithm_type} is not implemented for {self.__class__}"
                )

        # Validate and fix solver_type
        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                solver_type = "midpoint"
            else:
                raise NotImplementedError(
                    f"{solver_type} is not implemented for {self.__class__}"
                )

        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"] and final_sigmas_type == "zero":
            raise ValueError(
                f"`final_sigmas_type` {final_sigmas_type} is not supported for "
                f"`algorithm_type` {algorithm_type}. Please choose `sigma_min` instead."
            )

        # Store config as plain attributes
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.euler_at_final = euler_at_final
        self.final_sigmas_type = final_sigmas_type
        self.lambda_min_clipped = lambda_min_clipped
        self.variance_type = variance_type
        self.invert_sigmas = invert_sigmas

        # Setable values
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = sigmas.astype(np.float32)

        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = mx.array(sigmas)
        self.timesteps = self.sigmas * num_train_timesteps

        self.model_outputs: List[Optional[mx.array]] = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None

        self.sigma_min = float(self.sigmas[-1])
        self.sigma_max = float(self.sigmas[0])

    @property
    def step_index(self) -> Optional[int]:
        """The index counter for current timestep."""
        return self._step_index

    @property
    def begin_index(self) -> Optional[int]:
        """The index for the first timestep."""
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        """Sets the begin index for the scheduler."""
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        shift: Optional[float] = None,
    ) -> None:
        """Sets the discrete timesteps used for the diffusion chain.

        Args:
            num_inference_steps: Total number of time step spacings.
            sigmas: Custom sigma schedule.
            mu: Dynamic shifting parameter.
            shift: Shift factor override.
        """
        if self.use_dynamic_shifting and mu is None:
            raise ValueError(
                "you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`"
            )

        if sigmas is None:
            sigmas = np.linspace(
                self.sigma_max, self.sigma_min, num_inference_steps + 1
            ).copy()[:-1]

        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.final_sigmas_type == "zero":
            sigma_last = 0
        elif self.final_sigmas_type == "sigma_min":
            raise ValueError("sigma_min final_sigmas_type requires alphas_cumprod which is not available")
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.final_sigmas_type}"
            )

        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(timesteps).astype(mx.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0

        self._step_index = None
        self._begin_index = None

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def time_shift(self, mu: float, sigma: float, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def convert_model_output(
        self,
        model_output: mx.array,
        sample: mx.array,
    ) -> mx.array:
        """Convert model output to the type needed by the DPM-Solver algorithm.

        Args:
            model_output: Direct output from the learned diffusion model.
            sample: Current sample from the diffusion process.

        Returns:
            Converted model output.
        """
        # DPM-Solver++ needs to solve an integral of the data prediction model.
        if self.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be `flow_prediction`"
                )

            if self.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        # DPM-Solver needs to solve an integral of the noise prediction model.
        elif self.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.prediction_type} must be `flow_prediction`"
                )

            if self.thresholding:
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = model_output + x0_pred

            return epsilon

    def _threshold_sample(self, sample: mx.array) -> mx.array:
        """Dynamic thresholding to prevent pixel saturation."""
        batch_size, channels, *remaining_dims = sample.shape
        sample = sample.reshape(batch_size, channels * int(np.prod(remaining_dims)))

        abs_sample = mx.abs(sample)
        # Approximate quantile using sort
        sorted_vals = mx.sort(abs_sample, axis=1)
        idx = int(self.dynamic_thresholding_ratio * sorted_vals.shape[1])
        idx = min(idx, sorted_vals.shape[1] - 1)
        s = sorted_vals[:, idx:idx + 1]
        s = mx.clip(s, a_min=1.0, a_max=self.sample_max_value)
        sample = mx.clip(sample, a_min=-s, a_max=s) / s

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        return sample

    def dpm_solver_first_order_update(
        self,
        model_output: mx.array,
        sample: mx.array,
        noise: Optional[mx.array] = None,
    ) -> mx.array:
        """One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output: Direct output from the learned diffusion model.
            sample: Current sample from the diffusion process.
            noise: Optional noise for SDE variants.

        Returns:
            The sample tensor at the previous timestep.
        """
        sigma_t, sigma_s = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s = mx.log(alpha_s) - mx.log(sigma_s)

        h = lambda_t - lambda_s
        if self.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (mx.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (mx.exp(h) - 1.0)) * model_output
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * mx.exp(-h)) * sample
                + (alpha_t * (1 - mx.exp(-2.0 * h))) * model_output
                + sigma_t * mx.sqrt(1.0 - mx.exp(-2 * h)) * noise
            )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - 2.0 * (sigma_t * (mx.exp(h) - 1.0)) * model_output
                + sigma_t * mx.sqrt(mx.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[mx.array],
        sample: mx.array,
        noise: Optional[mx.array] = None,
    ) -> mx.array:
        """One step for the second-order multistep DPMSolver.

        Args:
            model_output_list: Model outputs at current and latter timesteps.
            sample: Current sample from the diffusion process.
            noise: Optional noise for SDE variants.

        Returns:
            The sample tensor at the previous timestep.
        """
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        lambda_s1 = mx.log(alpha_s1) - mx.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        if self.algorithm_type == "dpmsolver++":
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (mx.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (mx.exp(-h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (mx.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((mx.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.algorithm_type == "dpmsolver":
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (mx.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (mx.exp(h) - 1.0)) * D1
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (mx.exp(h) - 1.0)) * D0
                    - (sigma_t * ((mx.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * mx.exp(-h)) * sample
                    + (alpha_t * (1 - mx.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - mx.exp(-2.0 * h))) * D1
                    + sigma_t * mx.sqrt(1.0 - mx.exp(-2 * h)) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * mx.exp(-h)) * sample
                    + (alpha_t * (1 - mx.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - mx.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * mx.sqrt(1.0 - mx.exp(-2 * h)) * noise
                )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (mx.exp(h) - 1.0)) * D0
                    - (sigma_t * (mx.exp(h) - 1.0)) * D1
                    + sigma_t * mx.sqrt(mx.exp(2 * h) - 1.0) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (mx.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((mx.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * mx.sqrt(mx.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[mx.array],
        sample: mx.array,
    ) -> mx.array:
        """One step for the third-order multistep DPMSolver.

        Args:
            model_output_list: Model outputs at current and latter timesteps.
            sample: Current sample from the diffusion process.

        Returns:
            The sample tensor at the previous timestep.
        """
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
            self.sigmas[self.step_index - 2],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)

        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)
        lambda_s1 = mx.log(alpha_s1) - mx.log(sigma_s1)
        lambda_s2 = mx.log(alpha_s2) - mx.log(sigma_s2)

        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]

        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)

        if self.algorithm_type == "dpmsolver++":
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (mx.exp(-h) - 1.0)) * D0
                + (alpha_t * ((mx.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((mx.exp(-h) - 1.0 + h) / h ** 2 - 0.5)) * D2
            )
        elif self.algorithm_type == "dpmsolver":
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (mx.exp(h) - 1.0)) * D0
                - (sigma_t * ((mx.exp(h) - 1.0) / h - 1.0)) * D1
                - (sigma_t * ((mx.exp(h) - 1.0 - h) / h ** 2 - 0.5)) * D2
            )
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        if isinstance(timestep, mx.array):
            timestep = float(timestep)
        if isinstance(schedule_timesteps, mx.array):
            schedule_timesteps_np = np.array(schedule_timesteps.tolist())
        else:
            schedule_timesteps_np = np.array(schedule_timesteps)

        indices = np.where(schedule_timesteps_np == timestep)[0]
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos])

    def _init_step_index(self, timestep) -> None:
        """Initialize the step_index counter for the scheduler."""
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: mx.array,
        timestep: Union[int, mx.array],
        sample: mx.array,
        generator=None,
        variance_noise: Optional[mx.array] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict the sample from the previous timestep by reversing the SDE.

        Args:
            model_output: Direct output from learned diffusion model.
            timestep: Current discrete timestep in the diffusion chain.
            sample: Current sample from the diffusion process.
            generator: Random number generator (unused, for API compat).
            variance_noise: Alternative noise for SDE variants.
            return_dict: Whether to return a SchedulerOutput or tuple.

        Returns:
            SchedulerOutput or tuple with the previous sample.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.euler_at_final
            or (self.lower_order_final and len(self.timesteps) < 15)
            or self.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2)
            and self.lower_order_final
            and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast to float32 for precision
        sample = sample.astype(mx.float32)

        if self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
            noise = mx.random.normal(model_output.shape).astype(mx.float32)
        elif self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = variance_noise.astype(mx.float32)
        else:
            noise = None

        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=sample, noise=noise
            )
        elif self.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=sample, noise=noise
            )
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, sample=sample
            )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        # Cast sample back to expected dtype
        prev_sample = prev_sample.astype(model_output.dtype)

        # Upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: mx.array, *args, **kwargs) -> mx.array:
        """Ensures interchangeability with schedulers that need to scale input."""
        return sample

    def __len__(self) -> int:
        return self.num_train_timesteps
