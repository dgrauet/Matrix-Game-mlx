"""Cheap checks of the fixture generator's helpers (no weights, no model build)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
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


def test_realtime_steps_match_cli_resolution():
    """The generator's realtime preset (steps, codec) equals what the CLI's preset resolution gives for every scene;
    when the pack's inference.json carries realtime_steps / quality_steps, they must equal the resolved defaults."""
    from tests.conftest import weights_dir
    from tests.parity.gen_fixtures import PRESET_SPECS
    if weights_dir() is None:
        pytest.skip("MATRIX_GAME_4_MLX_WEIGHTS_DIR not set")
    from matrix_game_mlx.inference.loader import ModelComponents
    from matrix_game_mlx.scenes import model_args
    for scene in ("ancient-egypt", "castle", "racing"):
        comps = ModelComponents(model_args(scene, "realtime"), pack=weights_dir())
        assert (comps.args.steps, comps.args.vae_backend) == (PRESET_SPECS["realtime"].steps, "taehv")
        inference = json.loads((weights_dir() / comps.entry["inference"]).read_text())
        for key, preset in (("realtime_steps", "realtime"), ("quality_steps", "quality")):
            if key in inference:
                assert tuple(inference[key]) == model_args(scene, preset).steps, key


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


VARIANTS = {
    "racing_switch_realtime": ("racing", "taehv", (1000,), (1000,), True, 2),
    "racing_switch_quality": ("racing", "wan", (1000, 500), (1000, 500), True, 2),
    "castle_nosink_realtime": ("castle", "taehv", (1000,), (1000,), False, None),
    "racing_nosink_realtime": ("racing", "taehv", (1000,), (1000,), False, None),
    "castle_taehv_2step": ("castle", "taehv", (1000, 500), (1000, 500), True, None),
    "egypt_wan_custom_first": ("ancient-egypt", "wan", (1000,), (1000, 750, 500, 250), True, None),
    "racing_take_realtime": ("racing", "taehv", (1000,), (1000,), True, None),
    "racing_take_quality": ("racing", "wan", (1000, 500), (1000, 500), True, None),
    "racing_online_dynamic": ("racing", "taehv", (1000,), (1000,), True, None),
    "castle_online_realtime": ("castle", "taehv", (1000,), (1000,), True, None),
}


def test_variant_specs():
    """Exactly the registered variants, with their codec / steps / first steps / sink / prompt-switch chunk."""
    from tests.parity.gen_fixtures import VARIANT_SPECS
    assert set(VARIANT_SPECS) == set(VARIANTS)
    for name, (scene, backend, steps, first, sink, switch) in VARIANTS.items():
        v = VARIANT_SPECS[name]
        assert (v.name, v.scene, v.vae_backend, v.steps, v.first_steps, v.sink, v.switch_chunk) == \
            (name, scene, backend, steps, first, sink, switch)


def test_prompt_b_text():
    from tests.parity.gen_fixtures import PROMPT_B
    assert PROMPT_B == ("a racing game on a rainy city street at night, neon reflections on wet asphalt, "
                        "realistic AAA game rendering")


def test_variant_generation_commands():
    from tests.parity.gen_fixtures import VARIANT_SPECS, generation_command, parse_args
    for name in VARIANT_SPECS:
        assert generation_command(variant=name) == f"python -m tests.parity.gen_fixtures --variant {name}"
        assert parse_args(generation_command(variant=name).split()[3:]).variant == name
    assert generation_command(captions=True) == "python -m tests.parity.gen_fixtures --captions-only"
    assert generation_command("castle") == "python -m tests.parity.gen_fixtures --scene castle"   # unchanged


def test_print_variant_commands(capsys):
    """Captions first, then the six variants in registry order (the controller's background queue)."""
    from tests.parity.gen_fixtures import VARIANT_SPECS, main
    main(["--print-variant-commands"])
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["python -m tests.parity.gen_fixtures --captions-only"] + \
        [f"python -m tests.parity.gen_fixtures --variant {name}" for name in VARIANT_SPECS]


def test_variant_cli_validation():
    from tests.parity.gen_fixtures import parse_args
    args = parse_args(["--variant", "castle_taehv_2step", "--variant-floor-only"])
    assert (args.variant, args.variant_floor_only) == ("castle_taehv_2step", True)
    assert parse_args(["--captions-only"]).captions_only
    for argv in (["--variant-floor-only"], ["--variant", "nope"], ["--variant", "castle_taehv_2step", "--long"],
                 ["--variant", "castle_taehv_2step", "--floor-only"], ["--captions-only", "--variant",
                                                                        "castle_taehv_2step"]):
        with pytest.raises(SystemExit):
            parse_args(argv)


def test_variant_out_dir():
    from tests.conftest import fixtures_dir
    from tests.parity.gen_fixtures import VARIANT_SPECS, captions_dir, variant_dir
    for name in VARIANT_SPECS:
        assert variant_dir(name) == fixtures_dir() / name
    assert captions_dir() == fixtures_dir() / "captions"


def test_variant_denoising_schedule():
    """Upstream CastleStream.push step choice: first_frame uses first_steps on chunk 0 (none with a sink: the sink is
    the chunk), kv_only on positions < 2 (position = chunk + 1 with a sink)."""
    from tests.parity.gen_fixtures import (VARIANT_SPECS, variant_generator_calls, variant_n_renoise,
                                           variant_steps)
    egypt = VARIANT_SPECS["egypt_wan_custom_first"]
    assert [variant_steps(egypt, i) for i in range(3)] == [(1000, 750, 500, 250), (1000,), (1000,)]
    assert variant_generator_calls(egypt, 0) == 4 + 1                       # 4 denoising calls + cache update
    assert [variant_n_renoise(egypt, i) for i in range(3)] == [3, 0, 0]
    racing = VARIANT_SPECS["racing_switch_quality"]
    assert [len(variant_steps(racing, i)) for i in range(3)] == [0, 2, 2]
    assert [variant_generator_calls(racing, i) for i in range(3)] == [1, 3, 3]
    assert [variant_n_renoise(racing, i) for i in range(3)] == [0, 1, 1]
    assert [len(variant_steps(VARIANT_SPECS["racing_nosink_realtime"], i)) for i in range(3)] == [1, 1, 1]
    assert [len(variant_steps(VARIANT_SPECS["castle_nosink_realtime"], i)) for i in range(3)] == [1, 1, 1]
    assert [variant_n_renoise(VARIANT_SPECS["castle_taehv_2step"], i) for i in range(3)] == [1, 1, 1]


def test_variant_first_and_gated_chunks():
    from tests.parity.gen_fixtures import VARIANT_SPECS, gated_chunks
    assert gated_chunks(VARIANT_SPECS["racing_switch_realtime"]) == (1, 2)    # chunk 0 is the sink itself
    assert gated_chunks(VARIANT_SPECS["racing_switch_quality"]) == (1, 2)
    assert gated_chunks(VARIANT_SPECS["racing_nosink_realtime"]) == (0,)      # no sink: chunk 0 is denoised
    assert gated_chunks(VARIANT_SPECS["castle_nosink_realtime"]) == (0,)
    assert gated_chunks(VARIANT_SPECS["egypt_wan_custom_first"]) == (0,)


def test_variant_model_argv():
    """The MLX model arguments of a variant: every field explicit, so the preset default never decides."""
    from tests.parity.gen_fixtures import VARIANT_SPECS
    assert VARIANT_SPECS["castle_nosink_realtime"].model_argv() == \
        ["--vae-backend", "taehv", "--steps", "1000", "--first-steps", "1000", "--no-sink-image"]
    assert VARIANT_SPECS["egypt_wan_custom_first"].model_argv() == \
        ["--vae-backend", "wan", "--steps", "1000", "--first-steps", "1000,750,500,250"]
    assert VARIANT_SPECS["egypt_wan_custom_first"].preset == "quality"
    assert VARIANT_SPECS["racing_switch_realtime"].preset == "realtime"


def test_variant_model_args_resolve_to_spec():
    """Resolved through matrix_game_mlx.scenes.model_args exactly as the gates do."""
    from tests.conftest import weights_dir
    from tests.parity.gen_fixtures import VARIANT_SPECS
    if weights_dir() is None:
        pytest.skip("MATRIX_GAME_4_MLX_WEIGHTS_DIR not set")
    from matrix_game_mlx.scenes import model_args
    for v in VARIANT_SPECS.values():
        a = model_args(v.scene, v.preset, v.model_argv())
        assert (a.vae_backend, a.steps, a.first_steps, a.sink_image is not None) == \
            (v.vae_backend, v.steps, v.first_steps, v.sink), v.name


def test_variant_noise_draws_are_deterministic():
    """Initial noise per chunk and exactly variant_n_renoise renoise draws per chunk, reproducible."""
    from tests.parity.gen_fixtures import LATENT_SHAPE, VARIANT_SPECS, variant_n_renoise, variant_noises
    v = VARIANT_SPECS["egypt_wan_custom_first"]
    noises, renoises = variant_noises(v, 3)
    again = variant_noises(v, 3)
    assert len(noises) == 3 and all(n.shape == LATENT_SHAPE and n.dtype == np.float32 for n in noises)
    assert [len(r) for r in renoises] == [variant_n_renoise(v, i) for i in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(noises, again[0]))
    assert np.array_equal(renoises[0][2], again[1][0][2])


def test_caption_texts():
    from matrix_game_mlx.scenes import SCENES
    from tests.parity.gen_fixtures import PROMPT_B, caption_texts
    texts = caption_texts()
    assert texts == {"ancient-egypt": SCENES["ancient-egypt"]["prompt"], "castle": SCENES["castle"]["prompt"],
                     "racing": SCENES["racing"]["prompt"], "prompt_b": PROMPT_B}


def test_variant_inputs_round_trip(tmp_path):
    """inputs.npz as variant_reference writes it reads back through load_variant_inputs (sink / prompt B optional)."""
    from tests.parity.gen_fixtures import (LATENT_SHAPE, VARIANT_SPECS, load_variant_inputs, variant_inputs,
                                           variant_noises)
    pixels = np.zeros((9, 3, 4, 4), np.float32) + np.arange(9, dtype=np.float32)[:, None, None, None]
    for name, has_sink, has_b in (("racing_switch_quality", True, True), ("castle_nosink_realtime", False, False),
                                  ("egypt_wan_custom_first", True, False)):
        v = VARIANT_SPECS[name]
        noises, renoises = variant_noises(v)
        prompt, prompt_b = np.ones((1, 512, 8), np.float32), (np.full((1, 512, 8), 2, np.float32) if has_b else None)
        sink = np.full(LATENT_SHAPE, 3, np.float32) if has_sink else None
        np.savez(tmp_path / f"{name}.npz", **variant_inputs(prompt, prompt_b, sink, pixels, noises, renoises))
        got = load_variant_inputs(v, np.load(tmp_path / f"{name}.npz"))
        assert np.array_equal(got[0], pixels)
        assert all(np.array_equal(a, b) for a, b in zip(got[1], noises))
        assert [len(r) for r in got[2]] == [len(r) for r in renoises]
        assert all(np.array_equal(a, b) for ra, rb in zip(got[2], renoises) for a, b in zip(ra, rb))
        assert (got[3] is None) == (not has_sink) and (got[4] is None) == (not has_b)


class _FakeStream:
    """Mimics upstream CastleStream.push's generator / renoise calls (variant_steps per chunk + one clean pass)."""

    def __init__(self, v, consume_renoise=True):
        from tests.parity.gen_fixtures import LATENT_SHAPE, variant_steps
        self.v, self.index, self.events, self.device = v, 0, [], torch.device("cpu")
        self.steps, self.shape, self.consume = variant_steps, LATENT_SHAPE, consume_renoise
        self.gen = lambda **kw: (None, torch.zeros(self.shape))

    def set_prompt(self, embeddings):
        self.events.append(("set_prompt", self.index, embeddings.dtype))

    def push(self, pixels, *, frame_start, initial_noise=None):
        steps = self.steps(self.v, self.index)
        for k, _ in enumerate(steps):
            self.gen(noisy_image_or_video=initial_noise, timestep=torch.zeros(1, 1))
            if k + 1 < len(steps) and self.consume:
                self.events.append(("renoise", self.index, float(torch.randn(1, *self.shape[2:]).mean())))
        self.gen(noisy_image_or_video=initial_noise, timestep=torch.zeros(1, 1), drop_control=True)
        self.events.append(("push", self.index, frame_start, tuple(pixels.shape)))
        self.index += 1
        return "chunk"


def test_run_variant_switches_prompt_and_serves_renoise():
    from tests.parity.gen_fixtures import VARIANT_SPECS, run_variant, variant_noises
    pixels = np.zeros((9, 3, 2, 2), np.float32)
    v = VARIANT_SPECS["racing_switch_quality"]
    noises, renoises = variant_noises(v)
    stream, seen = _FakeStream(v), []
    run_variant(torch, v, stream, pixels, noises, renoises, np.zeros((1, 512, 8), np.float32),
                lambda i, chunk, preds, s: seen.append((i, len(preds))))
    assert seen == [(0, 0), (1, 2), (2, 2)]
    assert [e[:2] for e in stream.events] == [("push", 0), ("renoise", 1), ("push", 1), ("set_prompt", 2),
                                              ("renoise", 2), ("push", 2)]
    assert stream.events[3][2] == torch.bfloat16
    assert [e[2:] for e in stream.events if e[0] == "push"] == [(0, (1, 3, 2, 2)), (1, (4, 3, 2, 2)), (5, (4, 3, 2, 2))]
    # The served draw is the prepared one (renoise chunk 1 = renoises[1][0])
    assert stream.events[1][2] == pytest.approx(float(renoises[1][0].mean()), rel=1e-5)
    egypt = VARIANT_SPECS["egypt_wan_custom_first"]
    stream = _FakeStream(egypt)
    run_variant(torch, egypt, stream, pixels, *variant_noises(egypt), None, lambda *a: None)
    assert [e[:2] for e in stream.events].count(("renoise", 0)) == 3
    assert not any(e[0] == "set_prompt" for e in stream.events)
    with pytest.raises(RuntimeError, match="not consumed"):
        run_variant(torch, egypt, _FakeStream(egypt, consume_renoise=False), pixels, *variant_noises(egypt), None,
                    lambda *a: None)


def test_existing_variants_are_not_take_based():
    from tests.parity.gen_fixtures import VARIANT_SPECS
    assert {n for n, v in VARIANT_SPECS.items() if v.take} == {"racing_take_realtime", "racing_take_quality"}


def test_take_frames_shape_and_range():
    from tests import racing_game
    from tests.parity.gen_fixtures import take_frames
    take = racing_game.default_take()
    if not (take / "reference.mkv").is_file():
        pytest.skip(f"no scripted racing take at {take}")
    frames = take_frames(take, 9)
    assert frames.shape == (9, 3, 480, 864) and frames.dtype == np.float32
    assert frames.min() >= -1.0 and frames.max() <= 1.0
    assert not np.array_equal(frames[0], frames[8])     # the car is moving


# ---------------------------------------------------------------------------------------------------------------------
# Online captures (tests/parity/capture_online.py): real serve-mode game frames and engine captions.

def _write_states(capture, captions):
    """A synthetic states.jsonl in the capture_online layout: one row per chunk, the engine caption on the chunk's
    last state (as RacingGame sets it) and at the row's top level."""
    capture.mkdir(parents=True, exist_ok=True)
    rows, start = [], 0
    for i, caption in enumerate(captions):
        count = 1 if i == 0 else 4
        states = [dict(frame=f) for f in range(start, start + count)]
        states[-1]["chunk_prompt"] = caption
        rows.append(json.dumps(dict(frame_start=start, states=states, chunk_prompt=caption)))
        start += count
    (capture / "states.jsonl").write_text("\n".join(rows) + "\n")


def test_capture_prompts_replays_the_session_rule(tmp_path):
    """Session.produce (dynamic): on chunks with chunk % interval == 0 take states[-1]["chunk_prompt"] and switch when
    it differs from the current prompt, which starts at the scene profile prompt; None when unchanged."""
    from tests.parity.gen_fixtures import capture_prompts
    for captions, interval, expected in ((["A", "A", "B"], 1, ["A", None, "B"]),
                                         (["A", "C", "B"], 2, ["A", None, "B"]),     # chunk 1 is not sampled
                                         (["P", "P", "P"], 1, [None, None, None])):
        capture = tmp_path / "".join(captions) / str(interval)
        _write_states(capture, captions)
        assert capture_prompts(capture, 3, interval, "P") == expected, (captions, interval)
    _write_states(tmp_path / "short", ["A", "B"])
    with pytest.raises(ValueError, match="chunks"):
        capture_prompts(tmp_path / "short", 3, 1, "P")
    _write_states(tmp_path / "empty", ["A", ""])
    with pytest.raises(ValueError, match="caption"):
        capture_prompts(tmp_path / "empty", 2, 1, "P")


def _write_capture(capture, frames):
    from tests.parity.gen_fixtures import file_sha256
    capture.mkdir(parents=True, exist_ok=True)
    np.save(capture / "frames.npy", frames)
    (capture / "capture.json").write_text(json.dumps(dict(frames_sha256=file_sha256(capture / "frames.npy"))))


def test_capture_frames_returns_the_model_pixels_of_the_first_frames(tmp_path):
    """The first n captured RGB frames through frames_to_tensor (what Session.produce pushes), checked against the
    capture's recorded frames_sha256."""
    from PIL import Image
    from matrix_game_mlx.inference.preprocessing import frames_to_tensor
    from tests.conftest import to_np
    from tests.parity.gen_fixtures import capture_frames
    frames = np.random.default_rng(0).integers(0, 256, (5, 480, 864, 3), dtype=np.uint8)
    _write_capture(tmp_path / "ok", frames)
    got = capture_frames(tmp_path / "ok", 3)
    assert got.shape == (3, 3, 480, 864) and got.dtype == np.float32
    assert np.array_equal(got, to_np(frames_to_tensor([Image.fromarray(f) for f in frames[:3]])))
    with pytest.raises(ValueError, match="frames"):
        capture_frames(tmp_path / "ok", 6)
    _write_capture(tmp_path / "bad", frames)
    np.save(tmp_path / "bad" / "frames.npy", frames[::-1])                 # frames changed after the capture
    with pytest.raises(ValueError, match="frames_sha256"):
        capture_frames(tmp_path / "bad", 3)


def test_capture_dir_default_location():
    from tests.parity.gen_fixtures import capture_dir
    from tests.racing_game import resources_dir
    assert capture_dir("racing-online") == resources_dir() / "online-captures" / "racing-online"


def test_capture_variants():
    """The two online-capture variants, registered with their capture names; every other variant has none."""
    from tests.parity.gen_fixtures import VARIANT_SPECS, gated_chunks, generation_command
    assert {n: v.capture for n, v in VARIANT_SPECS.items() if v.capture} == {
        "racing_online_dynamic": "racing-online", "castle_online_realtime": "castle-online"}
    assert not any(VARIANT_SPECS[n].take for n in ("racing_online_dynamic", "castle_online_realtime"))
    assert generation_command(variant="racing_online_dynamic") == \
        "python -m tests.parity.gen_fixtures --variant racing_online_dynamic"
    # Every denoised chunk of the dynamic-prompt capture is gated (racing chunk 0 is the sink latent itself); the
    # fixed-prompt castle capture keeps the standing first-denoised-chunk rule
    assert gated_chunks(VARIANT_SPECS["racing_online_dynamic"]) == (1, 2)
    assert gated_chunks(VARIANT_SPECS["castle_online_realtime"]) == (0,)
    assert VARIANT_SPECS["racing_online_dynamic"].dynamic_prompts
    assert not VARIANT_SPECS["castle_online_realtime"].dynamic_prompts
    assert not VARIANT_SPECS["racing_switch_realtime"].dynamic_prompts


def test_chunk_prompts_per_chunk():
    """Per chunk, the embeddings set_prompt receives before the push: switch variants at switch_chunk, capture
    variants where capture_prompts switches."""
    from tests.parity.gen_fixtures import VARIANT_SPECS, chunk_prompts
    b = np.ones((1, 512, 8), np.float32)
    assert [e is b for e in chunk_prompts(VARIANT_SPECS["racing_switch_realtime"], b, 3)] == [False, False, True]
    assert chunk_prompts(VARIANT_SPECS["racing_switch_realtime"], b, 3)[:2] == [None, None]
    assert chunk_prompts(VARIANT_SPECS["castle_taehv_2step"], None, 3) == [None, None, None]
    per_chunk = [b, None, 2 * b]
    assert chunk_prompts(VARIANT_SPECS["racing_online_dynamic"], per_chunk, 3) is per_chunk


def test_capture_variant_inputs_round_trip(tmp_path):
    """Per-chunk prompt embeddings of a dynamic capture variant survive inputs.npz; a castle capture has none."""
    from tests.parity.gen_fixtures import (LATENT_SHAPE, VARIANT_SPECS, load_variant_inputs, variant_inputs,
                                           variant_noises)
    pixels = np.zeros((9, 3, 4, 4), np.float32)
    sink = np.zeros(LATENT_SHAPE, np.float32)
    for name, per_chunk in (("racing_online_dynamic", [np.full((1, 512, 8), 2, np.float32), None,
                                                       np.full((1, 512, 8), 3, np.float32)]),
                            ("castle_online_realtime", [None, None, None])):
        v = VARIANT_SPECS[name]
        noises, renoises = variant_noises(v)
        np.savez(tmp_path / f"{name}.npz", **variant_inputs(np.ones((1, 512, 8), np.float32), per_chunk, sink,
                                                            pixels, noises, renoises))
        got = load_variant_inputs(v, np.load(tmp_path / f"{name}.npz"))[4]
        assert [g is None for g in got] == [p is None for p in per_chunk]
        assert all(np.array_equal(g, p) for g, p in zip(got, per_chunk) if p is not None)


def test_run_variant_replays_capture_prompts():
    from tests.parity.gen_fixtures import VARIANT_SPECS, run_variant, variant_noises
    v = VARIANT_SPECS["racing_online_dynamic"]
    stream = _FakeStream(v)
    per_chunk = [np.zeros((1, 512, 8), np.float32), None, np.ones((1, 512, 8), np.float32)]
    run_variant(torch, v, stream, np.zeros((9, 3, 2, 2), np.float32), *variant_noises(v), per_chunk, lambda *a: None)
    assert [e[:2] for e in stream.events if e[0] == "set_prompt"] == [("set_prompt", 0), ("set_prompt", 2)]


def test_gated_caption_change():
    """The capture check: an engine-caption change between two consecutive gated chunks after chunk 0."""
    from tests.parity.capture_online import gated_caption_change
    assert gated_caption_change(["P", "A", "B"], (1, 2)) == 2
    assert gated_caption_change(["P", "A", "A"], (1, 2)) is None          # only the chunk-0 replacement (not gated)
    assert gated_caption_change(["P", "A", "A", "B"], (1, 2)) is None     # chunk 3 is not gated
    assert gated_caption_change(["A", "B", "B"], (0, 1, 2)) == 1


def test_capture_online_cli():
    from tests.parity.capture_online import parse_args
    from tests.parity.gen_fixtures import capture_dir
    args = parse_args(["--scene", "racing"])
    assert (args.scene, args.out) == ("racing", capture_dir("racing-online"))
    assert parse_args(["--scene", "castle", "--out", "/x"]).out == Path("/x")
    with pytest.raises(SystemExit):
        parse_args(["--scene", "ancient-egypt"])
