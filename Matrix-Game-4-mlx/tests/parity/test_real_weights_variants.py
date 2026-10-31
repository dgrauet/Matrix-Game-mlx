"""Real-weight parity of the model-path variants (prompt switch, no sink, free codec / step composition) against
upstream PyTorch fixtures (tests/parity/gen_fixtures.py --variant NAME, VARIANT_SPECS).

One test per variant whose fixture set exists ($MG4_FIXTURES_DIR/<variant>/, realtime layout, 3 chunks); a missing
or partial set is skipped with its gen_fixtures command. The MLX stream is built from the resolved CLI arguments of
the variant (matrix_game_mlx.scenes.model_args with the variant's explicit --vae-backend / --steps / --first-steps /
--no-sink-image) and fed torch's prompt embeddings, sink latent (variants with a sink), initial noises and renoise
draws; switch variants call CastleStream.set_prompt(torch's prompt B embeddings) before switch_chunk, as the torch
run did on the upstream stream. Online-capture variants (racing_online_dynamic, castle_online_realtime) push the
pixels of a real serve-mode capture (tests/parity/capture_online.py); their fixture's capture_frames_sha256 must
match the capture on disk, and racing_online_dynamic calls set_prompt(torch's embeddings of the engine caption)
before every chunk where the replayed upstream Session.produce rule switches (gen_fixtures.chunk_prompts).

Gated chunks (gen_fixtures.gated_chunks): the first denoised chunk (chunk 0, or chunk 1 for racing with a sink, whose
chunk 0 is the sink latent itself: no denoising, pred bitwise the sink), plus switch_chunk for switch variants; every
denoised chunk for racing_online_dynamic (an engine-caption switch can fall on any chunk; at least one must fall
between two gated chunks). Per gated chunk, by the standing floor method (floor measured per configuration,
gen_fixtures.variant_floor): control and layer 0 / 29 K cache rel L2 < max(2e-2, ceil_1sig(1.5 x floor)) (control floor: meta.json
floor.control_rel_l2; K floor: floor_chunk_i.npz vs the fixture), pred rel L2 < max(2e-2, 1.5 x floor), decoded PSNR
> 35 dB. The 2e-2 lower bound is the realtime suite's flat spec threshold (a floor-derived gate never goes below it). The other
chunks are reported (-s). Every chunk's denoising-call count must match torch's and gen_fixtures.variant_steps.

One variant's MLX components are held at a time (module-scoped `loaded`).
"""
import gc
import json
import sys

import numpy as np
import pytest
import mlx.core as mx

from tests.conftest import psnr, rel_l2, select_runtime, to_np
from tests.parity.gen_fixtures import (ONLINE_PROMPT_INTERVAL, VARIANT_SPECS, capture_command, capture_dir,
                                       capture_prompts, chunk_prompts, gated_chunks, generation_command,
                                       load_variant_inputs, variant_dir, variant_steps)
from tests.parity.test_real_weights import ceil_1sig, linear_cache

pytestmark = [pytest.mark.needs_fixtures, pytest.mark.needs_weights]

VARIANTS = tuple(VARIANT_SPECS)
KEYS = ("control", "pred", "kv_k_layer0", "kv_k_layer29")


def _meta(name: str):
    """The variant's meta.json if its set is complete (end to end and floor), else None."""
    path = variant_dir(name) / "meta.json"
    meta = json.loads(path.read_text()) if path.is_file() else None
    ok = meta and "control_rel_l2" in meta.get("floor", {}) and all(
        (variant_dir(name) / f).is_file() for f in ("inputs.npz", "chunks.npz",
                                                    *(f"floor_chunk_{i}.npz" for i in range(meta["chunks"]))))
    return meta if ok else None


def _require(name: str) -> dict:
    meta = _meta(name)
    if meta is None:
        pytest.skip(f"{variant_dir(name)} missing or incomplete (needs inputs.npz, chunks.npz, floor_chunk_*.npz, "
                    f"meta.json floor.control_rel_l2): generate it with {generation_command(variant=name)}")
    v = VARIANT_SPECS[name]
    assert (meta["variant"], meta["scene"], meta["vae_backend"], tuple(meta["steps"]), tuple(meta["first_steps"]),
            meta["sink"], meta["switch_chunk"], meta["prompt_b"]) == \
        (v.name, v.scene, v.vae_backend, v.steps, v.first_steps, v.sink, v.switch_chunk, v.prompt_b), meta
    if v.take:
        assert meta.get("take_reference_sha256"), \
            f"{name}: fixture lacks take_reference_sha256 (not generated from a take): regenerate it"
    if v.capture is not None:
        capture = capture_dir(v.capture)
        if not (capture / "capture.json").is_file():
            pytest.skip(f"no online capture at {capture}: record it with {capture_command(v)}")
        on_disk = json.loads((capture / "capture.json").read_text())["frames_sha256"]
        assert meta.get("capture_name") == v.capture and meta.get("capture_frames_sha256") == on_disk, \
            f"{name}: fixture capture {meta.get('capture_frames_sha256')} != {capture} {on_disk}: regenerate it"
        if v.dynamic_prompts:
            from matrix_game_mlx.scenes import SCENES
            assert meta["capture_prompts"] == capture_prompts(capture, meta["chunks"], ONLINE_PROMPT_INTERVAL,
                                                              SCENES[v.scene]["prompt"])
    return meta


@pytest.fixture(scope="module")
def loaded():
    """The one variant held in memory (its MLX components); see test_real_weights.loaded."""
    state = {}
    yield state
    state.clear()
    gc.collect()
    mx.clear_cache()


@pytest.fixture
def comps(variant, loaded, request):
    from matrix_game_mlx.inference.loader import ModelComponents
    from matrix_game_mlx.scenes import model_args
    meta = _require(variant)
    v = VARIANT_SPECS[variant]
    if loaded.get("variant") != variant:
        loaded.clear()
        for name in ("last_type", "last_value", "last_traceback", "last_exc"):
            # A failed test's postmortem traceback would keep the previous variant's DiT alive while this one loads
            sys.__dict__.pop(name, None)
        gc.collect()
        mx.clear_cache()
        inputs = np.load(variant_dir(variant) / "inputs.npz")
        comps = ModelComponents(select_runtime(model_args(v.scene, v.preset, v.model_argv()), request.config))
        a = comps.args
        assert (a.sink_mode, a.sink_rope_mode) == (meta["sink_mode"], meta["sink_rope_mode"])
        assert (a.vae_backend, a.steps, a.first_steps) == (v.vae_backend, v.steps, v.first_steps)
        assert (a.sink_image is not None) == v.sink == ("sink_latent" in inputs.files)
        comps.load()
        comps.prompt = mx.array(inputs["prompt_embeds"]).astype(mx.bfloat16)        # same embeddings as the torch run
        if v.sink:
            comps.sink_latent = mx.array(inputs["sink_latent"]).astype(mx.bfloat16)  # torch's sink
        assert (comps.sink_latent is None) == (not v.sink)
        loaded.update(variant=variant, comps=comps)
    return loaded["comps"]


class _StepRecorder:
    """Count the control-conditioned generator calls of a chunk."""

    def __init__(self, gen):
        self.gen, self.n = gen, 0

    def __call__(self, **kw):
        self.n += not kw.get("drop_control")
        return self.gen(**kw)


def _tolerances(meta: dict, floor: dict, ref, i: int) -> dict:
    """Gate of each quantity on chunk i from the torch floor of this configuration (module docstring)."""
    f = dict(control=meta["floor"]["control_rel_l2"][i], pred=rel_l2(floor["pred"], ref[f"pred_{i}"]),
             kv_k_layer0=rel_l2(floor["kv_k_layer0"], ref[f"kv_k_layer0_{i}"]),
             kv_k_layer29=rel_l2(floor["kv_k_layer29"], ref[f"kv_k_layer29_{i}"]))
    tol = {k: max(2e-2, ceil_1sig(1.5 * f[k])) for k in ("control", "kv_k_layer0", "kv_k_layer29")}
    tol["pred"] = max(2e-2, 1.5 * f["pred"])
    return f, tol


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_end_to_end(variant, comps):
    """Gated chunks by the floor rule; other chunks reported (module docstring)."""
    v, d, meta = VARIANT_SPECS[variant], variant_dir(variant), _meta(variant)
    inputs, ref = np.load(d / "inputs.npz"), np.load(d / "chunks.npz")
    _, noises, renoises, _, prompt_b = load_variant_inputs(v, inputs)
    prompts = chunk_prompts(v, prompt_b, meta["chunks"])
    first_frame = comps.args.sink_mode == "first_frame"
    gated = gated_chunks(v)
    if v.dynamic_prompts:
        assert [p is not None for p in prompts] == [p is not None for p in meta["capture_prompts"]]
        assert any(prompts[k] is not None for k in gated if k - 1 in gated), \
            f"{variant}: no engine-caption switch between two gated chunks {gated}: recapture"
    stream = comps.stream(seed=0)
    try:
        recorder = _StepRecorder(stream.gen)
        stream.gen = recorder
        failures, frame = [], 0
        for i in range(meta["chunks"]):
            if prompts[i] is not None:
                stream.set_prompt(mx.array(prompts[i]).astype(mx.bfloat16))
            px = mx.array(inputs[f"pixels_{i}"])
            recorder.n = 0
            c = stream.push(px, frame_start=frame, initial_noise=mx.array(noises[i]),
                            renoise=[mx.array(r) for r in renoises[i]])
            frame += px.shape[0]
            got = dict(control=to_np(c.control), pred=to_np(c.latent), kv_k_layer0=to_np(linear_cache(comps.gen, stream.kv[0])["k"]),
                       kv_k_layer29=to_np(linear_cache(comps.gen, stream.kv[29])["k"]))
            assert recorder.n == int(ref[f"n_denoising_{i}"]) == len(variant_steps(v, i)), \
                (i, recorder.n, int(ref[f"n_denoising_{i}"]), variant_steps(v, i))
            if first_frame and v.sink and i == 0:
                # No denoising call on either side: pred is the sink latent itself
                np.testing.assert_array_equal(got["pred"], to_np(comps.sink_latent), err_msg="chunk 0 pred vs sink latent")
                np.testing.assert_array_equal(got["pred"], ref["pred_0"], err_msg="chunk 0 pred vs torch")
            m = {k: rel_l2(got[k], ref[f"{k}_{i}"]) for k in KEYS}
            m["decoded_psnr"] = psnr(c.pixels, ref[f"decoded_{i}"])
            print(f"{variant} chunk {i}{'' if i in gated else ' (info)'}: " + ", ".join(f"{k}={x:.4g}" for k, x in m.items()))
            if i in gated:
                floor, tol = _tolerances(meta, np.load(d / f"floor_chunk_{i}.npz"), ref, i)
                print(f"{variant} chunk {i} torch floors: " + ", ".join(f"{k}={x:.3e} (tol {tol[k]:.3g})"
                                                                        for k, x in floor.items()))
                failures += [(i, k, m[k], tol[k]) for k in KEYS if not m[k] < tol[k]]
                if not m["decoded_psnr"] > 35.0:
                    failures.append((i, "decoded_psnr", m["decoded_psnr"], 35.0))
    finally:
        stream.close()
    assert not failures, failures
