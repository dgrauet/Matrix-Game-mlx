"""Cheap checks of the fixture generator's helpers (no weights, no model build)."""
import importlib.util
import sys

import pytest
import torch

from tests.parity.gen_fixtures import ensure_accelerate


@pytest.mark.skipif(importlib.util.find_spec("accelerate") is not None, reason="accelerate installed: no shim")
def test_ensure_accelerate_is_reentrant():
    """The default generation path builds the DiT several times in one process (fixtures, floors, calls, blocks)."""
    try:
        ensure_accelerate(torch)
        ensure_accelerate(torch)
        with sys.modules["accelerate"].init_empty_weights():
            layer = torch.nn.Linear(4, 4)
        assert layer.weight.device.type == "meta" and torch.nn.Linear(4, 4).weight.device.type == "cpu"
    finally:
        sys.modules.pop("accelerate", None)


def test_scene_specs():
    from tests.parity.gen_fixtures import SCENE_SPECS
    assert set(SCENE_SPECS) == {"ancient-egypt", "castle", "racing"}
    assert (SCENE_SPECS["racing"].sink_mode, SCENE_SPECS["racing"].sink_rope_mode) == ("first_frame", "absolute")
    assert SCENE_SPECS["racing"].src_dir == "driving_game"
    assert (SCENE_SPECS["castle"].sink_mode, SCENE_SPECS["castle"].sink_rope_mode) == ("kv_only", "fixed")
    assert SCENE_SPECS["castle"].src_dir == "fps-castle" and SCENE_SPECS["castle"].fixture_dir == "castle_quality"
    assert SCENE_SPECS["ancient-egypt"].fixture_dir == "egypt_quality"
    assert SCENE_SPECS["ancient-egypt"].checkpoint.parts[-2:] == ("fps-egypt", "model.safetensors")


def test_scene_out_dir():
    from tests.conftest import fixtures_dir
    from tests.parity.gen_fixtures import parse_args, scene_out_dir
    assert scene_out_dir(parse_args(["--scene", "castle", "--calls-only"])) == fixtures_dir() / "castle_quality"
    assert scene_out_dir(parse_args([])) == fixtures_dir() / "egypt_quality"


def test_first_denoising_call():
    from tests.parity.gen_fixtures import first_denoising_call
    names = [("calls_chunk0", 1), ("calls_chunk1", 3)]
    assert first_denoising_call(names, {"calls_chunk0": [True], "calls_chunk1": [False, False, True]}) == ("calls_chunk1", 0)
    names = [("calls_chunk0", 3), ("calls_chunk1", 3)]
    assert first_denoising_call(names, {"calls_chunk0": [False, False, True], "calls_chunk1": [False] * 3}) == ("calls_chunk0", 0)


def test_expected_denoising_calls_per_chunk():
    from tests.parity.gen_fixtures import SCENE_SPECS, STEPS, n_denoising_calls
    assert n_denoising_calls(SCENE_SPECS["racing"], 0) == 0
    assert n_denoising_calls(SCENE_SPECS["racing"], 1) == len(STEPS)
    assert n_denoising_calls(SCENE_SPECS["castle"], 0) == len(STEPS)


def test_renoise_count_follows_scene_policy():
    from tests.parity.gen_fixtures import SCENE_SPECS, STEPS, n_renoise
    assert [n_renoise(SCENE_SPECS["racing"], i) for i in (0, 1, 2, 22)] == [0, len(STEPS) - 1, len(STEPS) - 1, len(STEPS) - 1]
    assert [n_renoise(SCENE_SPECS["castle"], i) for i in (0, 1)] == [len(STEPS) - 1] * 2
    assert n_renoise(SCENE_SPECS["ancient-egypt"], 0) == 1


def test_queued_renoise_is_consumed_per_chunk():
    """Racing chunk 0 queues nothing (upstream draws no renoise), other chunks queue one tensor that must be drawn."""
    import numpy as np
    from tests.parity.gen_fixtures import LATENT_SHAPE, SCENE_SPECS, injected_renoise, n_renoise
    spec = SCENE_SPECS["racing"]
    for chunk, drawn in ((0, False), (1, True)):
        tensors = [torch.from_numpy(np.zeros(LATENT_SHAPE, np.float32))][:n_renoise(spec, chunk)]
        with injected_renoise(torch, tensors) as left:
            if drawn:
                torch.randn(1, *LATENT_SHAPE[2:])
        assert not left


def test_sink_rows_written_follows_upstream_rule():
    """k_sink_raw is written iff the slot starts inside the 405-token sink (racing chunk 0: slot 0 of an empty cache)."""
    from tests.parity.gen_fixtures import sink_rows_written
    assert sink_rows_written(405, 405) == range(0, 405)          # racing chunk-0 clean pass / kv_only sink priming
    assert not sink_rows_written(810, 405)                        # egypt chunk 0 (slot 1 after the primed sink)
    assert not sink_rows_written(21 * 405, 405)                   # full rolled window
    assert sink_rows_written(600, 405) == range(195, 405)         # a slot straddling the sink boundary


def test_call_recorder_stores_k_sink_raw_only_for_sink_writes():
    """Layer 0/29 k_sink_raw after the call is recorded (ksr{layer}) only when the call writes into the sink."""
    from tests.parity.gen_fixtures import KV_LAYERS, CallRecorder, from_bits
    kv = [dict(k=torch.zeros(1, 810, 2, 4, dtype=torch.bfloat16), v=torch.zeros(1, 810, 2, 4, dtype=torch.bfloat16),
               k_sink_raw=torch.zeros(1, 405, 2, 4, dtype=torch.bfloat16),
               global_end_index=torch.tensor([0]), local_end_index=torch.tensor([0])) for _ in range(30)]

    def gen(**kw):                                                # writes one 405-token slot at current_start
        start = kw["current_start"]
        for c in kv:
            c["k"][:, start:start + 405] = 1.0
            if start < 405:
                c["k_sink_raw"][:] = 2.0
            c["global_end_index"].fill_(start + 405)
            c["local_end_index"].fill_(start + 405)
        return None, torch.zeros(1, dtype=torch.bfloat16)

    recorder = CallRecorder(gen, kv)
    for start in (0, 405):
        recorder(noisy_image_or_video=torch.zeros(1, dtype=torch.bfloat16), timestep=torch.zeros(1, 1),
                 current_start=start, drop_control=True)
    first, second = recorder.calls
    assert all((from_bits(first[f"ksr{layer}"]) == 2.0).all() for layer in KV_LAYERS)
    assert not any(key.startswith("ksr") for key in second)


def test_preset_out_dir():
    """--preset realtime writes <scene-short>_realtime; the default preset keeps <scene-short>_quality."""
    from tests.conftest import fixtures_dir
    from tests.parity.gen_fixtures import SCENE_SPECS, parse_args, scene_out_dir
    assert parse_args([]).preset == "quality"
    assert scene_out_dir(parse_args(["--preset", "realtime", "--scene", "racing"])) == fixtures_dir() / "racing_realtime"
    assert scene_out_dir(parse_args(["--scene", "racing"])) == fixtures_dir() / "racing_quality"
    assert scene_out_dir(parse_args(["--preset", "realtime"])) == fixtures_dir() / "egypt_realtime"
    assert [SCENE_SPECS[s].preset_dir("realtime") for s in ("ancient-egypt", "castle", "racing")] == \
        ["egypt_realtime", "castle_realtime", "racing_realtime"]
    assert all(SCENE_SPECS[s].preset_dir("quality") == SCENE_SPECS[s].fixture_dir for s in SCENE_SPECS)
    with pytest.raises(SystemExit):
        parse_args(["--preset", "fast"])


def test_realtime_preset_spec():
    """Realtime: one denoising step at t=1000 (first chunk too), TAEHV codec, never a renoise draw."""
    from tests.parity.gen_fixtures import PRESET_SPECS, SCENE_SPECS, STEPS, n_denoising_calls, n_renoise
    realtime, quality = PRESET_SPECS["realtime"], PRESET_SPECS["quality"]
    assert (realtime.steps, realtime.vae_backend) == ((1000,), "taehv")
    assert (quality.steps, quality.vae_backend) == (STEPS, "wan")
    for spec in SCENE_SPECS.values():
        assert [n_renoise(spec, i, realtime.steps) for i in range(25)] == [0] * 25
    assert n_denoising_calls(SCENE_SPECS["racing"], 0, realtime.steps) == 0
    assert n_denoising_calls(SCENE_SPECS["racing"], 1, realtime.steps) == 1
    assert n_denoising_calls(SCENE_SPECS["castle"], 0, realtime.steps) == 1


def test_realtime_steps_match_pack_config():
    """The generator's realtime steps are the pack's inference.json realtime_steps (what ModelComponents uses)."""
    from tests.conftest import weights_dir
    from tests.parity.gen_fixtures import PRESET_SPECS
    if weights_dir() is None:
        pytest.skip("MATRIX_GAME_4_MLX_WEIGHTS_DIR not set")
    from matrix_game_mlx.inference.loader import ModelComponents
    for scene in ("ancient-egypt", "castle", "racing"):
        comps = ModelComponents(scene, "realtime", pack=weights_dir())
        assert (comps.steps, comps.vae_backend) == (PRESET_SPECS["realtime"].steps, "taehv")


def test_generation_command():
    from tests.parity.gen_fixtures import generation_command
    assert generation_command("racing", "realtime") == \
        "python -m tests.parity.gen_fixtures --scene racing --preset realtime"
    assert generation_command("castle") == "python -m tests.parity.gen_fixtures --scene castle"


def test_taehv_floor_summary():
    """Per quantity: max rel L2 over the torch variants (and min PSNR for decoded frames)."""
    import numpy as np
    from tests.conftest import psnr, rel_l2
    from tests.parity.gen_fixtures import taehv_floor_summary
    ref = np.linspace(-1, 1, 64, dtype=np.float32).reshape(4, 16)
    variants = {"fp32_cpu": ref + 0.01, "bf16_mps": ref - 0.02}
    row = taehv_floor_summary(ref, variants, with_psnr=True)
    assert row["fp32_cpu"] == pytest.approx(rel_l2(ref + 0.01, ref))
    assert row["max"] == pytest.approx(rel_l2(ref - 0.02, ref))
    assert row["min_psnr"] == pytest.approx(psnr(ref - 0.02, ref))
    assert "min_psnr" not in taehv_floor_summary(ref, variants)
    with pytest.raises(ValueError):
        taehv_floor_summary(ref, {})


def test_device_bridge_moves_latents_to_the_codec():
    """A DiT on another device hands its pred to the TAEHV decoder: the bridge moves it, nothing else changes."""
    from tests.parity.gen_fixtures import DeviceBridge

    class Codec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.seen, self.cleared = [], 0

        def encode_stream(self, x):
            self.seen.append(("enc", x.device, x.dtype))
            return x

        def decode_stream(self, z):
            self.seen.append(("dec", z.device, z.dtype))
            return z

        def clear_cache(self):
            self.cleared += 1

    codec = Codec()
    bridge = DeviceBridge(codec)
    assert next(bridge.parameters()) is codec.w
    bridge.clear_cache()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    bridge.decode_stream(torch.ones(2, dtype=torch.float32, device=device))
    bridge.encode_stream(torch.ones(1, dtype=torch.bfloat16))
    assert codec.seen == [("dec", torch.device("cpu"), torch.float32), ("enc", torch.device("cpu"), torch.bfloat16)]
    assert codec.cleared == 1
