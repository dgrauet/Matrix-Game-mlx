"""Validation tests for flow matching solvers (MLX vs PyTorch)."""
import sys
import os
import importlib.util
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PYTORCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Matrix-Game-3'))

import mlx.core as mx
from wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_module_from_path(name, filepath):
    """Load a Python module directly from a file path, bypassing sys.modules cache."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_torch_solvers():
    """Import PyTorch solvers for comparison using direct file loading."""
    torch_fm = _load_module_from_path(
        "torch_fm_solvers",
        os.path.join(PYTORCH_ROOT, "wan", "utils", "fm_solvers.py"),
    )
    torch_fm_unipc = _load_module_from_path(
        "torch_fm_solvers_unipc",
        os.path.join(PYTORCH_ROOT, "wan", "utils", "fm_solvers_unipc.py"),
    )
    return (
        torch_fm.FlowDPMSolverMultistepScheduler,
        torch_fm_unipc.FlowUniPCMultistepScheduler,
        torch_fm.get_sampling_sigmas,
    )


# ---------------------------------------------------------------------------
# Test 1: Sigma schedule
# ---------------------------------------------------------------------------

def test_sigma_schedule():
    """Verify sigma schedule matches for shift=5.0, steps=50."""
    sigmas_mlx = get_sampling_sigmas(50, 5.0)

    _, _, torch_get_sigmas = _get_torch_solvers()
    sigmas_torch = torch_get_sigmas(50, 5.0)

    np.testing.assert_allclose(
        sigmas_mlx, sigmas_torch, atol=1e-7, rtol=1e-7,
        err_msg="Sigma schedule mismatch"
    )
    print(f"  sigma_schedule: shape={sigmas_mlx.shape}, first={sigmas_mlx[0]:.6f}, last={sigmas_mlx[-1]:.6f}")


def test_sigma_schedule_various_shifts():
    """Verify sigma schedule for various shift values."""
    _, _, torch_get_sigmas = _get_torch_solvers()
    for shift in [1.0, 3.0, 7.5, 17.0]:
        for steps in [10, 20, 50]:
            sigmas_mlx = get_sampling_sigmas(steps, shift)
            sigmas_torch = torch_get_sigmas(steps, shift)
            np.testing.assert_allclose(
                sigmas_mlx, sigmas_torch, atol=1e-7, rtol=1e-7,
                err_msg=f"Mismatch for shift={shift}, steps={steps}"
            )


# ---------------------------------------------------------------------------
# Test 2: DPM Solver step
# ---------------------------------------------------------------------------

def test_dpm_solver_step():
    """Initialize DPM scheduler, run one step, compare output."""
    import torch

    TorchDPM, _, _ = _get_torch_solvers()

    np.random.seed(42)
    sample_np = np.random.randn(1, 4, 2, 4, 4).astype(np.float32)
    model_output_np = np.random.randn(1, 4, 2, 4, 4).astype(np.float32)

    sigmas = get_sampling_sigmas(20, 5.0)

    # MLX scheduler
    mlx_sched = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=1000, solver_order=2, shift=5.0,
    )
    mlx_sched.set_timesteps(sigmas=sigmas)

    sample_mx = mx.array(sample_np)
    model_out_mx = mx.array(model_output_np)
    timestep_mx = mlx_sched.timesteps[0]

    result_mx = mlx_sched.step(model_out_mx, timestep_mx, sample_mx)
    prev_mx = np.array(result_mx.prev_sample)

    # PyTorch scheduler
    torch_sched = TorchDPM(
        num_train_timesteps=1000, solver_order=2, shift=5.0,
    )
    torch_sched.set_timesteps(sigmas=sigmas)

    sample_pt = torch.tensor(sample_np)
    model_out_pt = torch.tensor(model_output_np)
    timestep_pt = torch_sched.timesteps[0]

    result_pt = torch_sched.step(model_out_pt, timestep_pt, sample_pt)
    prev_pt = result_pt.prev_sample.detach().cpu().numpy()

    np.testing.assert_allclose(
        prev_mx, prev_pt, atol=1e-5, rtol=1e-5,
        err_msg="DPM solver step mismatch"
    )
    print(f"  dpm_step: max_diff={np.max(np.abs(prev_mx - prev_pt)):.2e}")


# ---------------------------------------------------------------------------
# Test 3: UniPC Solver step
# ---------------------------------------------------------------------------

def test_unipc_solver_step():
    """Initialize UniPC scheduler, run one step, compare output."""
    import torch

    _, TorchUniPC, _ = _get_torch_solvers()

    np.random.seed(123)
    sample_np = np.random.randn(1, 4, 2, 4, 4).astype(np.float32)
    model_output_np = np.random.randn(1, 4, 2, 4, 4).astype(np.float32)

    sigmas = get_sampling_sigmas(20, 5.0)

    # MLX scheduler
    mlx_sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, solver_order=2, shift=5.0,
    )
    mlx_sched.set_timesteps(sigmas=sigmas)

    sample_mx = mx.array(sample_np)
    model_out_mx = mx.array(model_output_np)
    timestep_mx = mlx_sched.timesteps[0]

    result_mx = mlx_sched.step(model_out_mx, timestep_mx, sample_mx)
    prev_mx = np.array(result_mx.prev_sample)

    # PyTorch scheduler
    torch_sched = TorchUniPC(
        num_train_timesteps=1000, solver_order=2, shift=5.0,
    )
    torch_sched.set_timesteps(sigmas=sigmas)

    sample_pt = torch.tensor(sample_np)
    model_out_pt = torch.tensor(model_output_np)
    timestep_pt = torch_sched.timesteps[0]

    result_pt = torch_sched.step(model_out_pt, timestep_pt, sample_pt)
    prev_pt = result_pt.prev_sample.detach().cpu().numpy()

    np.testing.assert_allclose(
        prev_mx, prev_pt, atol=1e-5, rtol=1e-5,
        err_msg="UniPC solver step mismatch"
    )
    print(f"  unipc_step: max_diff={np.max(np.abs(prev_mx - prev_pt)):.2e}")


# ---------------------------------------------------------------------------
# Test 4: Full denoise loop
# ---------------------------------------------------------------------------

def test_full_denoise_loop():
    """Run a full denoise loop with both schedulers, compare final samples."""
    import torch

    TorchDPM, TorchUniPC, _ = _get_torch_solvers()

    num_steps = 20
    shape = (1, 4, 2, 4, 4)
    sigmas = get_sampling_sigmas(num_steps, 5.0)

    # Generate deterministic "model outputs" for each step
    np.random.seed(999)
    model_outputs = [np.random.randn(*shape).astype(np.float32) for _ in range(num_steps)]
    initial_sample = np.random.randn(*shape).astype(np.float32)

    for SchedulerMLX, SchedulerPT, name in [
        (FlowDPMSolverMultistepScheduler, TorchDPM, "DPM"),
        (FlowUniPCMultistepScheduler, TorchUniPC, "UniPC"),
    ]:
        # MLX loop
        mlx_sched = SchedulerMLX(num_train_timesteps=1000, solver_order=2, shift=5.0)
        mlx_sched.set_timesteps(sigmas=sigmas)
        sample_mx = mx.array(initial_sample.copy())
        for i, t in enumerate(mlx_sched.timesteps):
            out = mx.array(model_outputs[i])
            result = mlx_sched.step(out, t, sample_mx)
            sample_mx = result.prev_sample

        final_mx = np.array(sample_mx)

        # PyTorch loop
        pt_sched = SchedulerPT(num_train_timesteps=1000, solver_order=2, shift=5.0)
        pt_sched.set_timesteps(sigmas=sigmas)
        sample_pt = torch.tensor(initial_sample.copy())
        for i, t in enumerate(pt_sched.timesteps):
            out = torch.tensor(model_outputs[i])
            result = pt_sched.step(out, t, sample_pt)
            sample_pt = result.prev_sample

        final_pt = sample_pt.detach().cpu().numpy()

        max_diff = np.max(np.abs(final_mx - final_pt))
        np.testing.assert_allclose(
            final_mx, final_pt, atol=1e-4, rtol=1e-4,
            err_msg=f"{name} full denoise loop mismatch"
        )
        print(f"  {name}_full_loop: max_diff={max_diff:.2e}")


# ---------------------------------------------------------------------------
# Test 5: set_timesteps consistency
# ---------------------------------------------------------------------------

def test_set_timesteps_consistency():
    """Verify set_timesteps produces matching sigmas and timesteps."""
    import torch

    TorchDPM, TorchUniPC, _ = _get_torch_solvers()
    sigmas = get_sampling_sigmas(30, 3.0)

    for SchedulerMLX, SchedulerPT, name in [
        (FlowDPMSolverMultistepScheduler, TorchDPM, "DPM"),
        (FlowUniPCMultistepScheduler, TorchUniPC, "UniPC"),
    ]:
        mlx_s = SchedulerMLX(num_train_timesteps=1000, solver_order=2, shift=3.0)
        mlx_s.set_timesteps(sigmas=sigmas)

        pt_s = SchedulerPT(num_train_timesteps=1000, solver_order=2, shift=3.0)
        pt_s.set_timesteps(sigmas=sigmas)

        mlx_sigmas = np.array(mlx_s.sigmas)
        pt_sigmas = pt_s.sigmas.numpy()

        np.testing.assert_allclose(
            mlx_sigmas, pt_sigmas, atol=1e-6, rtol=1e-6,
            err_msg=f"{name} sigmas after set_timesteps mismatch"
        )
        print(f"  {name}_set_timesteps: sigmas match, len={len(mlx_sigmas)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
