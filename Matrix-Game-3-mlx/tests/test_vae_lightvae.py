"""Validation tests for the MG-LightVAE decoder path (vae_type='mg_lightvae').

The reference (Matrix-Game-3/generate.py) defaults to --vae_type
mg_lightvae_v2: a Wan2.2 VAE whose decoder is a Turbo-VAED student pruned at
rate 0.75, paired with the full ("teacher") Wan2.2 VAE encoder for
conditioning latents. Decode goes through the pruned student, encode through
the teacher.

Weights: the mlx-forge converted repo ships vae_lightvae_v2.safetensors
(student, prefix 'vae_lightvae_v2.') alongside vae.safetensors (teacher).
Parity is checked against the official MG-LightVAE_v2.pth loaded by the
PyTorch reference implementation.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlx.core as mx

from wan.modules.vae2_2 import Wan2_2_VAE, WanVAE_

_SNAPSHOT = os.path.expanduser("~/Work/mlx/models/matrix-game-3.0-mlx-local")
LIGHTVAE_V2_SFT = os.path.join(_SNAPSHOT, "vae_lightvae_v2.safetensors")
FULL_VAE_SFT = os.path.join(_SNAPSHOT, "vae.safetensors")
LIGHTVAE_V2_PTH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Skywork--Matrix-Game-3.0/"
    "snapshots"
)

_REFERENCE_VAE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..",
        "Matrix-Game-3", "wan", "modules", "vae2_2.py",
    )
)

needs_weights = pytest.mark.skipif(
    not (os.path.exists(LIGHTVAE_V2_SFT) and os.path.exists(FULL_VAE_SFT)),
    reason="converted VAE weights not in HF cache",
)


def _find_official_pth():
    if not os.path.isdir(LIGHTVAE_V2_PTH):
        return None
    for root, _dirs, files in os.walk(LIGHTVAE_V2_PTH):
        if "MG-LightVAE_v2.pth" in files:
            return os.path.join(root, "MG-LightVAE_v2.pth")
    return None


class TestLightVaeWrapper:
    @needs_weights
    def test_ctor_builds_teacher_encoder_and_pruned_decoder(self):
        vae = Wan2_2_VAE(
            z_dim=48,
            c_dim=160,
            dec_dim=256,
            vae_pth=LIGHTVAE_V2_SFT,
            dtype=mx.float32,
            vae_type="mg_lightvae",
            lightvae_pruning_rate=0.75,
            lightvae_encoder_vae_pth=FULL_VAE_SFT,
        )
        assert vae.encoder_model is not None
        # Student decoder pruned at 0.75: conv1 out = 1024 * 0.25 = 256
        assert vae.model.decoder.conv1.weight.shape[0] == 256
        # Teacher encoder is the full VAE
        assert vae.encoder_model.encoder.conv1.weight.shape[0] == 160

    @needs_weights
    def test_decode_output_shape(self):
        vae = Wan2_2_VAE(
            z_dim=48,
            c_dim=160,
            dec_dim=256,
            vae_pth=LIGHTVAE_V2_SFT,
            dtype=mx.float32,
            vae_type="mg_lightvae",
            lightvae_pruning_rate=0.75,
            lightvae_encoder_vae_pth=FULL_VAE_SFT,
        )
        z = mx.random.normal((1, 4, 4, 48)).astype(mx.float32)
        out = vae.decode([z])[0]
        mx.eval(out)
        assert out.shape == (1, 64, 64, 3)


class TestLoadVaeLightVae:
    @needs_weights
    def test_load_vae_resolves_lightvae_v2_from_directory(self):
        from pipeline.vae_config import load_vae

        vae = load_vae(
            model_path=_SNAPSHOT,
            vae_type="mg_lightvae_v2",
            dtype=mx.float32,
        )
        assert vae.encoder_model is not None
        assert vae.model.decoder.conv1.weight.shape[0] == 256


class TestLightVaeDecodeParity:
    @needs_weights
    def test_decode_matches_pytorch_reference(self):
        pth = _find_official_pth()
        if pth is None:
            pytest.skip("official MG-LightVAE_v2.pth not in HF cache")

        spec = importlib.util.spec_from_file_location("ref_vae2_2", _REFERENCE_VAE)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        import torch

        z_np = np.random.default_rng(42).standard_normal(
            (1, 48, 1, 4, 4)
        ).astype("float32")

        # PyTorch reference: student decoder from the official checkpoint
        ref_model = ref._video_vae(
            pretrained_path=pth,
            z_dim=48,
            dim=160,
            dim_mult=[1, 2, 4, 4],
            temperal_downsample=[False, True, True],
            pruning_rate=0.75,
        ).eval()
        mean_pt = torch.tensor(Wan2_2_VAE.MEAN, dtype=torch.float32)
        inv_std_pt = 1.0 / torch.tensor(Wan2_2_VAE.STD, dtype=torch.float32)
        with torch.no_grad():
            out_pt = ref_model.decode(
                torch.from_numpy(z_np), [mean_pt, inv_std_pt]
            ).clamp_(-1, 1)
        out_pt_np = out_pt.squeeze(0).permute(1, 2, 3, 0).numpy()  # (T, H, W, C)

        # MLX: converted weights through the wrapper
        vae = Wan2_2_VAE(
            z_dim=48,
            c_dim=160,
            dec_dim=256,
            vae_pth=LIGHTVAE_V2_SFT,
            dtype=mx.float32,
            vae_type="mg_lightvae",
            lightvae_pruning_rate=0.75,
            lightvae_encoder_vae_pth=FULL_VAE_SFT,
        )
        z_mx = mx.array(np.transpose(z_np[0], (1, 2, 3, 0)))  # (T, h, w, C)
        out_mx = vae.decode([z_mx])[0]
        mx.eval(out_mx)
        out_mx_np = np.array(out_mx)

        assert out_mx_np.shape == out_pt_np.shape
        max_abs = np.max(np.abs(out_mx_np - out_pt_np))
        assert max_abs < 1e-4, f"LightVAE decode diverges: max_abs={max_abs}"


class TestGenerateVaeTypeArg:
    def test_default_vae_type_mirrors_reference(self, monkeypatch):
        import generate

        monkeypatch.setattr(
            sys, "argv",
            ["generate.py", "--prompt", "p", "--image", "i.png"],
        )
        args = generate._parse_args()
        assert args.vae_type == "mg_lightvae_v2"
