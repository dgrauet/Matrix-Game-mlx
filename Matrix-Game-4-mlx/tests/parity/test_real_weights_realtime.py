"""Real-weight parity of the realtime preset (TAEHV taew2_2 codec, one denoising step) against upstream PyTorch
fixtures (tests/parity/gen_fixtures.py --preset realtime).

Every test runs once per scene whose realtime fixture set exists (ancient-egypt -> egypt_realtime, castle ->
castle_realtime, racing -> racing_realtime); a missing or partial set is skipped with its gen_fixtures command. The
DiT is preset-independent and gated per block / per call by the quality suite (test_real_weights.py); here:

- TAEHV sink latent: the pack's sink image through the MLX TAEHV (encode_sink_image, scene sink mode) vs torch's,
  rel L2 below 1.5 x the torch variant floor (meta.json "taehv_floors" sink: max rel L2 of the fp32 CPU / bf16 MPS
  torch variants vs the bf16 CPU reference), rounded up to one significant digit (ceil_1sig).
- TAEHV calls: on fresh streaming state, MLX encode_stream of each chunk's fixture pixels (control) and MLX
  decode_stream of each chunk's torch end-to-end pred (teacher forced, decoded), chunk by chunk; each chunk's control
  and decoded rel L2 below ceil_1sig(1.5 x that chunk's torch variant floor); decoded PSNR reported.
- End to end: the realtime MLX stream fed torch's sink latent, prompt embeddings and initial noises. Only the first
  denoised chunk is gated with the spec thresholds (control / pred / layer 0 and 29 K cache rel L2 < 2e-2, decoded
  PSNR > 35 dB): chunk 0, plus chunk 1 for racing, whose chunk 0 is the sink latent itself (no denoising, pred bitwise
  the sink) and whose chunk-1 pred gate is 1.5 x the torch floor of that chunk (floor_chunk_1.npz, the M2 rule). The
  other chunks are reported (-s).

One scene's MLX DiT is held at a time (tests/conftest.py groups this module's items by scene); the quality module's
DiT is freed by its own module-scoped fixture before this module loads.
"""
import gc
import json
import sys

import numpy as np
import pytest
import mlx.core as mx

from tests.conftest import fixtures_dir, psnr, rel_l2, to_np
from tests.parity.gen_fixtures import PRESET_SPECS, SCENE_SPECS, generation_command
from tests.parity.test_real_weights import ceil_1sig

pytestmark = [pytest.mark.needs_fixtures, pytest.mark.needs_weights]

SCENES = tuple(SCENE_SPECS)                    # ancient-egypt, castle, racing
PRESET = "realtime"


def _dir(scene: str):
    return fixtures_dir() / SCENE_SPECS[scene].preset_dir(PRESET)


def _meta(scene: str, *keys: str):
    """The scene's realtime meta.json if it exists and has every key in `keys`, else None."""
    path = _dir(scene) / "meta.json"
    meta = json.loads(path.read_text()) if path.is_file() else None
    return meta if meta and all(k in meta for k in keys) else None


def _require(scene: str, *keys: str) -> dict:
    meta = _meta(scene, *keys)
    if meta is None:
        pytest.skip(f"{_dir(scene)} missing or incomplete (needs meta.json {', '.join(keys)}): generate it with "
                    f"{generation_command(scene, PRESET)}")
    assert meta["preset"] == PRESET and tuple(meta["steps"]) == PRESET_SPECS[PRESET].steps, meta
    return meta


@pytest.fixture(scope="module")
def taehv_cache():
    return {}


@pytest.fixture
def taehv(scene, taehv_cache):
    """The pack's MLX TAEHV in bf16, as ModelComponents loads it for the realtime preset (loaded once, after the
    scene's fixtures are found)."""
    from matrix_game_mlx.inference.loader import pack_dir
    from matrix_game_mlx.inference.taehv import TAEHV
    _require(scene, "taehv_floors")
    if "taehv" not in taehv_cache:
        taehv_cache["taehv"] = TAEHV.from_pack(pack_dir() / "taehv.safetensors", dtype=mx.bfloat16)
    return taehv_cache["taehv"]


@pytest.mark.parametrize("scene", SCENES)
def test_taehv_sink_latent(scene, taehv):
    """The pack's sink image through the MLX TAEHV vs torch's encode_sink_image (gen_fixtures inputs.npz)."""
    from matrix_game_mlx.inference.loader import ModelComponents
    from matrix_game_mlx.inference.stream import encode_sink_image
    meta = _require(scene, "taehv_floors")
    comps = ModelComponents(scene, PRESET)                                # config only, nothing loaded
    # The image torch encoded is <checkpoint dir>/sink.png (gen_fixtures SceneSpec); the pack's copy must be the same
    assert comps.sink_image == comps.pack / SCENE_SPECS[scene].src_dir / "sink.png"
    assert comps.sink_image.read_bytes() == SCENE_SPECS[scene].sink_image.read_bytes()
    torch_sink = np.load(_dir(scene) / "inputs.npz")["sink_latent"]
    sink = encode_sink_image(taehv, comps.sink_image, "taehv", first_frame=comps.sink_mode == "first_frame")
    floor = meta["taehv_floors"]["sink"]["max"]
    tol = ceil_1sig(1.5 * floor)
    err = rel_l2(sink, torch_sink)
    print(f"{scene} TAEHV sink rel L2 {err:.3e} tol {tol:.0e} (torch floor {floor:.3e})")
    assert err < tol, (err, tol)


@pytest.mark.parametrize("scene", SCENES)
def test_taehv_calls(scene, taehv):
    """Streaming encode of the fixture pixels and teacher-forced streaming decode of the torch preds, per chunk."""
    meta = _require(scene, "taehv_floors")
    inputs, calls, chunks = (np.load(_dir(scene) / f"{name}.npz") for name in ("inputs", "taehv_calls", "chunks"))
    floors = meta["taehv_floors"]["chunks"]
    taehv.clear_cache()
    failures = []
    for i, floor in enumerate(floors):
        x = mx.array(inputs[f"pixels_{i}"]).transpose(1, 0, 2, 3)[None].astype(taehv.dtype)   # [1, 3, T, H, W]
        control = taehv.encode_stream(x).transpose(0, 2, 1, 3, 4)                            # [1, 1, 48, h, w]
        pred = mx.array(chunks[f"pred_{i}"]).astype(mx.bfloat16)                             # torch end-to-end pred
        decoded = taehv.decode_stream(pred.transpose(0, 2, 1, 3, 4))[0].transpose(1, 0, 2, 3)  # [T, 3, H, W]
        tol = {k: ceil_1sig(1.5 * floor[k]["max"]) for k in ("control", "decoded")}
        m = dict(control=rel_l2(control, calls[f"control_{i}"]), decoded=rel_l2(decoded, calls[f"decoded_{i}"]))
        print(f"{scene} TAEHV chunk {i}: control {m['control']:.3e} (tol {tol['control']:.0e}), decoded "
              f"{m['decoded']:.3e} (tol {tol['decoded']:.0e}), decoded PSNR {psnr(decoded, calls[f'decoded_{i}']):.1f} dB "
              f"(torch variants min {floor['decoded']['min_psnr']:.1f} dB)")
        failures += [(i, k, m[k], tol[k]) for k in m if not m[k] < tol[k]]
    taehv.clear_cache()
    assert not failures, failures


@pytest.fixture(scope="module")
def loaded():
    """The one scene held in memory (its realtime MLX components); see test_real_weights.loaded."""
    state = {}
    yield state
    state.clear()
    gc.collect()
    mx.clear_cache()


@pytest.fixture
def comps(scene, loaded):
    from matrix_game_mlx.inference.loader import ModelComponents
    meta = _require(scene, "taehv_floors", "floor")
    if loaded.get("scene") != scene:
        loaded.clear()
        for name in ("last_type", "last_value", "last_traceback", "last_exc"):
            # A failed test's postmortem traceback would keep the previous scene's DiT alive while this one loads
            sys.__dict__.pop(name, None)
        gc.collect()
        mx.clear_cache()
        inputs = np.load(_dir(scene) / "inputs.npz")
        comps = ModelComponents(scene, PRESET)
        assert (comps.sink_mode, comps.sink_rope_mode) == (meta["sink_mode"], meta["sink_rope_mode"])
        assert (comps.vae_backend, comps.steps) == ("taehv", tuple(meta["steps"]))
        comps.load()
        comps.prompt = mx.array(inputs["prompt_embeds"]).astype(mx.bfloat16)     # same embeddings as the torch run
        comps.sink_latent = mx.array(inputs["sink_latent"]).astype(mx.bfloat16)  # torch's TAEHV sink
        loaded.update(scene=scene, comps=comps)
    return loaded["comps"]


class _StepRecorder:
    """Count the control-conditioned generator calls of a chunk."""

    def __init__(self, gen):
        self.gen, self.n = gen, 0

    def __call__(self, **kw):
        self.n += not kw.get("drop_control")
        return self.gen(**kw)


@pytest.mark.parametrize("scene", SCENES)
def test_stream_end_to_end_realtime(scene, comps):
    """Chunk 0 (and racing chunk 1) gated with the spec thresholds; other chunks reported (see module docstring)."""
    d, meta = _dir(scene), _meta(scene)
    inputs, ref = np.load(d / "inputs.npz"), np.load(d / "chunks.npz")
    stream = comps.stream(seed=0)
    recorder = _StepRecorder(stream.gen)
    stream.gen = recorder
    first_frame = comps.sink_mode == "first_frame"
    gated = {0, 1} if first_frame else {0}
    failures, frame = [], 0
    for i in range(meta["chunks"]):
        px = mx.array(inputs[f"pixels_{i}"])
        recorder.n = 0
        c = stream.push(px, frame_start=frame, initial_noise=mx.array(inputs[f"initial_noise_{i}"]))
        frame += px.shape[0]
        got = dict(control=to_np(c.control), pred=to_np(c.latent), kv_k_layer0=to_np(stream.kv[0]["k"]),
                   kv_k_layer29=to_np(stream.kv[29]["k"]))
        assert recorder.n == int(ref[f"n_denoising_{i}"]), (i, recorder.n, int(ref[f"n_denoising_{i}"]))
        if first_frame and i == 0:
            # No denoising call on either side: pred is the sink latent itself
            np.testing.assert_array_equal(got["pred"], to_np(comps.sink_latent), err_msg="chunk 0 pred vs sink latent")
            np.testing.assert_array_equal(got["pred"], ref["pred_0"], err_msg="chunk 0 pred vs torch")
        m = {k: rel_l2(v, ref[f"{k}_{i}"]) for k, v in got.items()}
        m["decoded_psnr"] = psnr(c.pixels, ref[f"decoded_{i}"])
        print(f"{scene} realtime end-to-end chunk {i}{'' if i in gated else ' (info)'}: "
              + ", ".join(f"{k}={v:.4g}" for k, v in m.items()))
        if i in gated:
            # Racing chunk 1 (first denoised chunk after the sink frame): pred gate 1.5 x the torch-vs-variant floor
            # of that chunk, as the quality suite (M2 rule)
            pred_tol = 2e-2 if i == 0 else 1.5 * rel_l2(np.load(d / f"floor_chunk_{i}.npz")["pred"], ref[f"pred_{i}"])
            checks = dict(control=(m["control"], 2e-2), pred=(m["pred"], pred_tol),
                          kv_k_layer0=(m["kv_k_layer0"], 2e-2), kv_k_layer29=(m["kv_k_layer29"], 2e-2))
            failures += [(i, k, v, tol) for k, (v, tol) in checks.items() if not v < tol]
            if not m["decoded_psnr"] > 35.0:
                failures.append((i, "decoded_psnr", m["decoded_psnr"], 35.0))
    assert not failures, failures
