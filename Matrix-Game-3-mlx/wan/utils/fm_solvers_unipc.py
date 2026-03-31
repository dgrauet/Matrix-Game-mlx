# Ported from Matrix-Game-3/wan/utils/fm_solvers_unipc.py
# Flow matching UniPC multistep scheduler — MLX port (no diffusers dependency)

import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from collections import namedtuple

SchedulerOutput = namedtuple("SchedulerOutput", ["prev_sample"])


class FlowUniPCMultistepScheduler:
    """UniPC multistep scheduler for flow matching.

    A training-free framework for fast sampling of diffusion models using the
    UniPC (Unified Predictor-Corrector) algorithm adapted for flow matching.

    Args:
        num_train_timesteps: Number of diffusion steps to train the model.
        solver_order: UniPC order (any positive integer). Recommended: 2 for guided, 3 for unconditional.
        prediction_type: Must be ``"flow_prediction"``.
        shift: Factor to adjust sigmas in the noise schedule.
        use_dynamic_shifting: Whether to apply dynamic shifting based on resolution.
        thresholding: Whether to use dynamic thresholding.
        dynamic_thresholding_ratio: Ratio for dynamic thresholding.
        sample_max_value: Threshold value for dynamic thresholding.
        predict_x0: Whether to use the updating algorithm on predicted x0.
        solver_type: Solver type (``"bh1"`` or ``"bh2"``).
        lower_order_final: Whether to use lower-order solvers in final steps.
        disable_corrector: Steps at which to disable the corrector.
        solver_p: Optional external predictor scheduler.
        timestep_spacing: How timesteps are scaled.
        steps_offset: Offset added to inference steps.
        final_sigmas_type: Final sigma value type (``"zero"`` or ``"sigma_min"``).
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
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: Optional[List[int]] = None,
        solver_p=None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: Optional[str] = "zero",
    ):
        if disable_corrector is None:
            disable_corrector = []

        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                solver_type = "bh2"
            else:
                raise NotImplementedError(
                    f"{solver_type} is not implemented for {self.__class__}"
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
        self.predict_x0 = predict_x0
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.final_sigmas_type = final_sigmas_type

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
        self.timestep_list: List[Optional[Union[int, mx.array]]] = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample: Optional[mx.array] = None
        self._step_index: Optional[int] = None
        self._begin_index: Optional[int] = None
        self.this_order: Optional[int] = None

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
        self.last_sample = None
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps)

        self._step_index = None
        self._begin_index = None

    def _sigma_to_t(self, sigma):
        return sigma * self.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def time_shift(self, mu: float, sigma: float, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def _threshold_sample(self, sample: mx.array) -> mx.array:
        """Dynamic thresholding to prevent pixel saturation."""
        batch_size, channels, *remaining_dims = sample.shape
        sample = sample.reshape(batch_size, channels * int(np.prod(remaining_dims)))

        abs_sample = mx.abs(sample)
        sorted_vals = mx.sort(abs_sample, axis=1)
        idx = int(self.dynamic_thresholding_ratio * sorted_vals.shape[1])
        idx = min(idx, sorted_vals.shape[1] - 1)
        s = sorted_vals[:, idx:idx + 1]
        s = mx.clip(s, a_min=1.0, a_max=self.sample_max_value)
        sample = mx.clip(sample, a_min=-s, a_max=s) / s

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        return sample

    def convert_model_output(
        self,
        model_output: mx.array,
        sample: mx.array,
    ) -> mx.array:
        """Convert model output to the type needed by the UniPC algorithm.

        Args:
            model_output: Direct output from the learned diffusion model.
            sample: Current sample from the diffusion process.

        Returns:
            Converted model output.
        """
        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
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
        else:
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

    def multistep_uni_p_bh_update(
        self,
        model_output: mx.array,
        sample: mx.array,
        order: int,
    ) -> mx.array:
        """One step for the UniP (B(h) version).

        Args:
            model_output: Direct output from the learned diffusion model.
            sample: Current sample from the diffusion process.
            order: The order of UniP at this timestep.

        Returns:
            The sample tensor at the previous timestep.
        """
        model_output_list = self.model_outputs

        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = mx.log(alpha_si) - mx.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = mx.array(rks)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = mx.exp(hh) - 1  # expm1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = mx.exp(hh) - 1  # expm1
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(mx.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = mx.stack(R)
        b = mx.array(b)

        if len(D1s) > 0:
            D1s_stacked = mx.stack(D1s, axis=1)  # (B, K, C, ...)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = mx.array([0.5]).astype(x.dtype)
            else:
                rhos_p = mx.linalg.solve(R[:-1, :-1], b[:-1], stream=mx.cpu).astype(x.dtype)
        else:
            D1s_stacked = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s_stacked is not None:
                pred_res = mx.einsum("k,bkc...->bc...", rhos_p, D1s_stacked)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s_stacked is not None:
                pred_res = mx.einsum("k,bkc...->bc...", rhos_p, D1s_stacked)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.astype(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: mx.array,
        last_sample: mx.array,
        this_sample: mx.array,
        order: int,
    ) -> mx.array:
        """One step for the UniC (B(h) version) corrector.

        Args:
            this_model_output: Model output at current timestep.
            last_sample: Sample before the last predictor step.
            this_sample: Sample after the last predictor step.
            order: The order of UniC-p (effective accuracy is order + 1).

        Returns:
            Corrected sample tensor.
        """
        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = mx.log(alpha_t) - mx.log(sigma_t)
        lambda_s0 = mx.log(alpha_s0) - mx.log(sigma_s0)

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = mx.log(alpha_si) - mx.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = mx.array(rks)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = mx.exp(hh) - 1  # expm1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.solver_type == "bh1":
            B_h = hh
        elif self.solver_type == "bh2":
            B_h = mx.exp(hh) - 1  # expm1
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(mx.power(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = mx.stack(R)
        b = mx.array(b)

        if len(D1s) > 0:
            D1s_stacked = mx.stack(D1s, axis=1)
        else:
            D1s_stacked = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = mx.array([0.5]).astype(x.dtype)
        else:
            rhos_c = mx.linalg.solve(R, b, stream=mx.cpu).astype(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s_stacked is not None:
                corr_res = mx.einsum("k,bkc...->bc...", rhos_c[:-1], D1s_stacked)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s_stacked is not None:
                corr_res = mx.einsum("k,bkc...->bc...", rhos_c[:-1], D1s_stacked)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        x_t = x_t.astype(x.dtype)
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
        return_dict: bool = True,
        generator=None,
    ) -> Union[SchedulerOutput, Tuple]:
        """Predict the sample from the previous timestep using UniPC.

        Args:
            model_output: Direct output from learned diffusion model.
            timestep: Current discrete timestep in the diffusion chain.
            sample: Current sample from the diffusion process.
            return_dict: Whether to return a SchedulerOutput or tuple.
            generator: Unused, for API compatibility.

        Returns:
            SchedulerOutput or tuple with the previous sample.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = (
            self.step_index > 0
            and self.step_index - 1 not in self.disable_corrector
            and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)

        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # pass the original non-converted model output
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

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
