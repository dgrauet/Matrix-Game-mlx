"""Real-weight gates of the apple presets (float16 ring-fused DiT; apple-fast: window 11) against the realtime
upstream PyTorch fixtures (tests/parity/gen_fixtures.py --preset realtime).

The DiT runs in float16 here, so the realtime suite's thresholds do not apply; these gates were fixed before the
first run (spec "Validation", apple presets):

- End to end (the realtime suite's loop: torch's sink latent, prompt embeddings and initial noises fed to the MLX
  stream, `--preset apple`). The gated chunks are the realtime suite's (castle and egypt chunk 0, racing chunks 0
  and 1), the first denoising calls on torch's exact inputs, so the spec's per-call and end-to-end gates are both
  checked there: pred rel L2 vs torch <= max(6e-2, 2 x the realtime pred gate of that chunk), the realtime gate
  being max(2e-2, 1.5 x the torch floor of the chunk) (floor_chunk_i.npz); decoded PSNR >= 30 dB; control rel L2
  < 2e-2 (TAEHV is bf16 and unchanged, the realtime gate). Layer 0 and 29 K-cache rel L2 are printed, not gated.
  Racing chunk 0 is the reference-first sink latent: its pred is bitwise `sink_latent.astype(float16)` (rel L2 vs
  torch reported).
- apple-fast: its 11-latent window acts only past position 11 and every fixture chunk is before it, so on the same
  loaded DiT every chunk's pred and decoded frames are bitwise the apple run's.
- fp16 safety: 32 free-running chunks per scene and apple runtime (past the first ring eviction at position 21 and
  past 22) on the scene sink image's pan (tests.compare_runtimes.Pan), stream seed 0, eager blocks (`use_graph`
  off; the compiled graph is bitwise eager, tests/test_runtime.py): every pred, `k`, `v` and `k_sink_raw` finite,
  and the residual stream's max |x| (the residual output of `ln_modulate`, `gate_ln_affine` and `gate_residual`)
  below 65504 / 16.

One scene's MLX DiT is held at a time (tests/conftest.py groups this module's items by scene); apple-fast reuses the
loaded DiT through `install_runtime`, so no second DiT is ever built.
"""
import gc
import sys

import numpy as np
import pytest
import mlx.core as mx

from tests.compare_runtimes import Pan
from tests.conftest import psnr, rel_l2, to_np
from tests.parity.gen_fixtures import SCENE_SPECS
from tests.parity.test_real_weights import linear_cache
from tests.parity.test_real_weights_realtime import _StepRecorder, _dir, _require

pytestmark = [pytest.mark.needs_fixtures, pytest.mark.needs_weights]

SCENES = tuple(SCENE_SPECS)                    # ancient-egypt, castle, racing
PRESET = "apple"
FP16_MAX = 65504.0
RESIDUAL_LIMIT = FP16_MAX / 16
SAFETY_CHUNKS = 32
CONTROL_TOL = 2e-2
PSNR_MIN = 30.0


def pred_gate(floor: float) -> float:
    """max(6e-2, 2 x the realtime pred gate max(2e-2, 1.5 x floor)) (spec "Validation", per call)."""
    return max(6e-2, 2 * max(2e-2, 1.5 * floor))


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


@pytest.fixture(scope="module")
def loaded():
    """The one scene held in memory (its apple MLX components) and its apple fixture run."""
    state = {}
    yield state
    state.clear()
    gc.collect()
    mx.clear_cache()


@pytest.fixture
def comps(scene, loaded):
    """The scene's components under `--preset apple` (float16 DiT), with torch's prompt embeddings and sink latent
    substituted as the realtime suite does (bf16, as the loader produces them)."""
    from matrix_game_mlx.inference.loader import ModelComponents
    from matrix_game_mlx.scenes import model_args
    meta = _require(scene, "taehv_floors", "floor")
    if loaded.get("scene") != scene:
        loaded.clear()
        for name in ("last_type", "last_value", "last_traceback", "last_exc"):
            # A failed test's postmortem traceback would keep the previous scene's DiT alive while this one loads
            sys.__dict__.pop(name, None)
        gc.collect()
        mx.clear_cache()
        inputs = np.load(_dir(scene) / "inputs.npz")
        comps = ModelComponents(model_args(scene, PRESET))
        assert (comps.args.dit_runtime, comps.args.vae_backend) == ("apple", "taehv")
        assert (comps.args.sink_mode, comps.args.sink_rope_mode) == (meta["sink_mode"], meta["sink_rope_mode"])
        assert comps.args.steps == tuple(meta["steps"])
        comps.load()
        assert comps.dit_runtime() == "apple" and comps.gen.model.patch_embedding.weight.dtype == mx.float16
        comps.prompt = mx.array(inputs["prompt_embeds"]).astype(mx.bfloat16)     # same embeddings as the torch run
        comps.sink_latent = mx.array(inputs["sink_latent"]).astype(mx.bfloat16)  # torch's TAEHV sink
        loaded.update(scene=scene, comps=comps)
    return loaded["comps"]


def _fixture_loop(scene, comps, metrics: bool):
    """The realtime end-to-end loop on the scene's fixtures under the installed runtime.

    Returns (rows, kept): per chunk the metrics vs torch (only with `metrics`; the K caches are compared in torch's
    21-latent linear layout) and the pred and decoded frames as float32 numpy (exact casts of float16 / bf16)."""
    d, meta = _dir(scene), _require(scene, "taehv_floors", "floor")
    inputs, ref = np.load(d / "inputs.npz"), np.load(d / "chunks.npz")
    stream = comps.stream(seed=0)
    rows, kept = [], []
    try:
        recorder = _StepRecorder(stream.gen)
        stream.gen = recorder
        frame = 0
        for i in range(meta["chunks"]):
            px = mx.array(inputs[f"pixels_{i}"])
            recorder.n = 0
            c = stream.push(px, frame_start=frame, initial_noise=mx.array(inputs[f"initial_noise_{i}"]))
            frame += px.shape[0]
            assert recorder.n == int(ref[f"n_denoising_{i}"]), (i, recorder.n, int(ref[f"n_denoising_{i}"]))
            assert c.latent.dtype == mx.float16, c.latent.dtype
            kept.append({"pred": to_np(c.latent), "decoded": to_np(c.pixels)})
            if metrics:
                m = dict(control=rel_l2(c.control, ref[f"control_{i}"]), pred=rel_l2(c.latent, ref[f"pred_{i}"]))
                for layer in (0, 29):
                    k = to_np(linear_cache(comps.gen, stream.kv[layer])["k"])
                    m[f"kv_k_layer{layer}"] = rel_l2(k, ref[f"kv_k_layer{layer}_{i}"])
                    del k
                m["decoded_psnr"] = psnr(c.pixels, ref[f"decoded_{i}"])
                rows.append(m)
            del c
    finally:
        stream.close()
    return rows, kept


@pytest.mark.parametrize("scene", SCENES)
def test_apple_end_to_end_gates(scene, comps, loaded):
    """The gated chunks (castle / egypt chunk 0, racing chunks 0 and 1) against the a-priori apple gates; every
    chunk's metrics printed (see the module docstring)."""
    d = _dir(scene)
    ref = np.load(d / "chunks.npz")
    first_frame = comps.args.sink_mode == "first_frame"
    gated = {0, 1} if first_frame else {0}
    rows, kept = _fixture_loop(scene, comps, metrics=True)
    loaded["apple_run"] = kept
    failures = []
    for i, m in enumerate(rows):
        floor = rel_l2(np.load(d / f"floor_chunk_{i}.npz")["pred"], ref[f"pred_{i}"])
        gate = pred_gate(floor)
        print(f"{scene} apple end-to-end chunk {i}{'' if i in gated else ' (info)'}: "
              + ", ".join(f"{k}={v:.4g}" for k, v in m.items()) + f", pred gate {gate:.4g} (torch floor {floor:.4g})")
        if first_frame and i == 0:
            # Reference-first chunk: no denoising call, the pred is the bf16 sink latent cast to the DiT dtype
            want = to_np(comps.sink_latent.astype(mx.float16))
            np.testing.assert_array_equal(_bits(kept[0]["pred"]), _bits(want), err_msg="chunk 0 pred vs fp16 sink")
        if i in gated:
            if not m["pred"] <= gate:
                failures.append((i, "pred", m["pred"], gate))
            if not m["control"] < CONTROL_TOL:
                failures.append((i, "control", m["control"], CONTROL_TOL))
            if not m["decoded_psnr"] >= PSNR_MIN:
                failures.append((i, "decoded_psnr", m["decoded_psnr"], PSNR_MIN))
    assert not failures, failures


@pytest.mark.parametrize("scene", SCENES)
def test_apple_fast_is_bitwise_apple_on_fixtures(scene, comps, loaded):
    """apple-fast on the same loaded DiT: every fixture chunk (all before position 11) bitwise the apple run."""
    from matrix_game_mlx.inference.runtime import install_runtime
    if "apple_run" not in loaded:                         # run alone (-k): the apple reference first
        loaded["apple_run"] = _fixture_loop(scene, comps, metrics=False)[1]
    apple = loaded["apple_run"]
    install_runtime(comps.gen, "apple-fast")
    try:
        assert comps.dit_runtime() == "apple-fast"
        _, fast = _fixture_loop(scene, comps, metrics=False)
        window = comps.gen.model.blocks[0].self_attn.local_attn_size       # set by the stream from the runtime
        assert window == 11, window
    finally:
        install_runtime(comps.gen, "apple")
    assert len(fast) == len(apple)
    for i, (a, b) in enumerate(zip(apple, fast)):
        for name in ("pred", "decoded"):
            np.testing.assert_array_equal(_bits(b[name]), _bits(a[name]), err_msg=f"chunk {i} {name}")
    print(f"{scene} apple-fast vs apple: {len(fast)} chunks, pred and decoded frames bitwise")


@pytest.fixture
def residual_max(monkeypatch):
    """Pending max |x| of the residual-stream outputs of the fused kernels (run_blocks imports them at call time)."""
    from matrix_game_mlx.inference import kernels
    pending = []

    def wrap(name, residual):
        original = getattr(kernels, name)

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            pending.append(mx.max(mx.abs(residual(out))))
            return out

        monkeypatch.setattr(kernels, name, wrapped)

    wrap("ln_modulate", lambda out: out[1])            # (z, x'): x' = x + residual, or x itself
    wrap("gate_ln_affine", lambda out: out[1])         # (z, x')
    wrap("gate_residual", lambda out: out)
    return pending


def _drain(pending) -> float:
    assert pending, "no fused kernel call recorded"
    value = float(mx.max(mx.stack(pending)).astype(mx.float32))
    pending.clear()
    return value


@pytest.mark.parametrize("runtime", ("apple", "apple-fast"))
@pytest.mark.parametrize("scene", SCENES)
def test_apple_fp16_safety(scene, runtime, comps, residual_max):
    """32 free-running chunks (eager blocks): finite preds and KV, residual stream max |x| < 65504 / 16."""
    from matrix_game_mlx.inference.runtime import install_runtime
    install_runtime(comps.gen, runtime)
    try:
        comps.gen.model.castle_runtime.use_graph = False
        pan = Pan(comps.args.sink_image)
        stream = comps.stream(seed=0)
        try:
            window = comps.gen.model.blocks[0].self_attn.local_attn_size
            assert window == (11 if runtime == "apple-fast" else 21), window
            maxima, positions, frame = [], [], 0
            creation = _drain(residual_max) if residual_max else 0.0   # kv_only sink call at creation
            assert creation < RESIDUAL_LIMIT, ("sink call", creation, RESIDUAL_LIMIT)
            for i in range(SAFETY_CHUNKS):
                count = 1 if i == 0 else 4
                pixels = mx.array(np.stack([pan.frame(frame + k) for k in range(count)]))
                positions.append(stream.index + stream.sink_offset)
                c = stream.push(pixels, frame_start=frame)
                frame += count
                finite = [mx.all(mx.isfinite(c.latent))] + [mx.all(mx.isfinite(cache[name])) for cache in stream.kv
                                                            for name in ("k", "v", "k_sink_raw")]
                mx.eval(finite)
                assert bool(finite[0]), f"chunk {i}: non-finite pred"
                bad = [j for j, f in enumerate(finite[1:]) if not bool(f)]
                assert not bad, f"chunk {i}: non-finite KV entries (layer, k/v/k_sink_raw): " \
                                f"{[(j // 3, j % 3) for j in bad]}"
                maxima.append(_drain(residual_max))
                assert maxima[-1] < RESIDUAL_LIMIT, (i, maxima[-1], RESIDUAL_LIMIT)
                del c, finite
        finally:
            stream.close()
    finally:
        install_runtime(comps.gen, "apple")
    worst = int(np.argmax(maxima))
    print(f"{scene} {runtime} fp16 safety: {SAFETY_CHUNKS} chunks, positions {positions[0]}-{positions[-1]}, "
          f"residual max |x| {max(maxima):.1f} at chunk {worst} (limit {RESIDUAL_LIMIT:.0f}; sink call "
          f"{creation:.1f}); mx peak {mx.get_peak_memory() / 1e9:.2f} GB; per chunk "
          + " ".join(f"{v:.0f}" for v in maxima))
