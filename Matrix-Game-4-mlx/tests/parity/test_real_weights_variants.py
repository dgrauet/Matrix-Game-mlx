"""Real-weight parity of the model-path variants (prompt switch, no sink, free codec / step composition) against
upstream PyTorch fixtures (tests/parity/gen_fixtures.py --variant NAME, VARIANT_SPECS).

One test per variant whose fixture set exists ($MG4_FIXTURES_DIR/<variant>/, realtime layout, 3 chunks); a missing
or partial set is skipped with its gen_fixtures command. The MLX stream is built from the resolved CLI arguments of
the variant (matrix_game_mlx.scenes.model_args with the variant's explicit --vae-backend / --steps / --first-steps /
--no-sink-image) and fed torch's prompt embeddings, sink latent (variants with a sink), initial noises and renoise
draws; switch variants call CastleStream.set_prompt(torch's prompt B embeddings) before switch_chunk, as the torch
run did on the upstream stream.

Gated chunks (gen_fixtures.gated_chunks): the first denoised chunk (chunk 0, or chunk 1 for racing with a sink, whose
chunk 0 is the sink latent itself: no denoising, pred bitwise the sink), plus switch_chunk for switch variants. Per
gated chunk, by the standing floor method (floor measured per configuration, gen_fixtures.variant_floor):
control and layer 0 / 29 K cache rel L2 < max(2e-2, ceil_1sig(1.5 x floor)) (control floor: meta.json
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

from tests.conftest import psnr, rel_l2, to_np
from tests.parity.gen_fixtures import (VARIANT_SPECS, gated_chunks, generation_command, load_variant_inputs,
                                       variant_dir, variant_steps)
from tests.parity.test_real_weights import ceil_1sig

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
def comps(variant, loaded):
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
        comps = ModelComponents(model_args(v.scene, v.preset, v.model_argv()))
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
    first_frame = comps.args.sink_mode == "first_frame"
    gated = gated_chunks(v)
    stream = comps.stream(seed=0)
    try:
        recorder = _StepRecorder(stream.gen)
        stream.gen = recorder
        failures, frame = [], 0
        for i in range(meta["chunks"]):
            if i == v.switch_chunk:
                stream.set_prompt(mx.array(prompt_b).astype(mx.bfloat16))
            px = mx.array(inputs[f"pixels_{i}"])
            recorder.n = 0
            c = stream.push(px, frame_start=frame, initial_noise=mx.array(noises[i]),
                            renoise=[mx.array(r) for r in renoises[i]])
            frame += px.shape[0]
            got = dict(control=to_np(c.control), pred=to_np(c.latent), kv_k_layer0=to_np(stream.kv[0]["k"]),
                       kv_k_layer29=to_np(stream.kv[29]["k"]))
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
