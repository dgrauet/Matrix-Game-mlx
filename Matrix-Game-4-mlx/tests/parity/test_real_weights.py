"""Real-weight parity of the MLX stream against upstream PyTorch fixtures (tests/parity/gen_fixtures.py).

Every test runs once per scene whose fixture set is complete (ancient-egypt -> egypt_quality, castle ->
castle_quality, racing -> racing_quality; a missing or partial set is skipped with its gen_fixtures command). Only
one scene's MLX DiT is loaded at a time. The scene's sink policy (meta.json "sink_mode" / "sink_rope_mode") drives
the MLX side: kv_only/fixed scenes prime the sink at construction, racing (first_frame/absolute) reuses the sink
latent as chunk 0 and denoises from chunk 1.

Three levels:
- Per block (isolated): for the first control-conditioned call (meta.json "block_call": chunk 0 call 0, racing
  chunk 1 call 0), each MLX block gets torch's exact block input (and shared e0 / context / sink KV); its output
  must be within 1.5 x the largest rel L2 between torch and its own kernel variants on that block (meta.json
  "block_floors"). This is the check that nothing diverges systematically.
- Per call (teacher forced): torch's exact cache state at a chunk boundary and torch's exact inputs are injected,
  one MLX generator call runs, and its pred plus the layer 0/29 KV it writes are compared with torch. Each call's
  threshold is 1.5 x the largest rel L2 between torch and its kernel variants on the same call (meta.json
  "call_floors"), rounded up to one significant digit (call_tol). The rest of the cache (history, sink, rolled
  window) and the end indices of every layer must match torch exactly; k_sink_raw too, except the rows a call writes
  when its slot starts inside the sink (racing's chunk-0 clean pass), which are gated like the KV slot on layers
  0/29. Long chunk 22 (needs `gen_fixtures --long`, skipped otherwise) runs past the 21-frame window, so the rolling
  eviction (and, for fixed sink RoPE, the re-RoPE) run with real weights.
- End to end: the full MLX stream from the same inputs. The autoregressive stream amplifies bf16 rounding chunk
  over chunk (torch against itself reaches ~0.3 rel L2 by chunk 2), so only the first denoised chunk is gated, with
  the spec thresholds: chunk 0, plus chunk 1 for racing, whose chunk 0 is the sink latent itself (no denoising,
  pred bitwise equal to the sink). The other chunks are reported.
"""
import gc
import json
import math

import numpy as np
import pytest
import mlx.core as mx

from tests.conftest import fixtures_dir, psnr, rel_l2, to_np
from tests.parity.gen_fixtures import FLOOR_VARIANTS, KV_LAYERS, SCENE_SPECS, from_bits, sink_rows_written, written_slot

pytestmark = pytest.mark.needs_fixtures
needs_weights = pytest.mark.needs_weights

SCENES = tuple(SCENE_SPECS)                    # ancient-egypt, castle, racing


def _dir(scene: str):
    return fixtures_dir() / SCENE_SPECS[scene].fixture_dir


def _meta(scene: str):
    """The scene's meta.json, or None while its fixture set is missing or partial (no call / block floors yet)."""
    path = _dir(scene) / "meta.json"
    meta = json.loads(path.read_text()) if path.is_file() else None
    return meta if meta and "call_floors" in meta and "block_floors" in meta else None


def _call_cases():
    """(scene, fixture name, call index) per recorded call; one placeholder per scene without fixtures (skipped)."""
    cases = []
    for scene in SCENES:
        meta = _meta(scene)
        if meta is None:
            cases.append((scene, None, None))
            continue
        calls = meta["call_floors"]["calls"]
        cases += [(scene, name, j) for name in sorted(calls) for j in range(len(calls[name]))]
    return cases


def ceil_1sig(x: float) -> float:
    """Round x >= 0 up to one significant digit (2.67e-2 -> 3e-2), as the exact decimal (3e-4, not 3.0000000000000004e-4)."""
    if x == 0:
        return 0.0
    e = math.floor(math.log10(x))
    d = math.ceil(round(x / 10 ** e, 9))       # 1.5 x 2e-4 must give 3e-4, not 4e-4
    return float(f"1e{e + 1}") if d == 10 else float(f"{d}e{e}")


def call_tol(floors: dict, t_zero: bool) -> dict:
    """A call's thresholds from its meta.json "call_floors" row: ceil_1sig(1.5 x max over the kernel variants) of pred
    ("pred"; 0 for t=0 calls, whose pred is the input exactly), of max(k, v) of the KV slot written at each KV layer
    ("l{layer}"), and of k_sink_raw ("ksr{layer}") when the call writes into the sink."""
    tol = {"pred": 0.0 if t_zero else ceil_1sig(1.5 * max(floors[f"{v}_pred"] for v in FLOOR_VARIANTS))}
    for layer in KV_LAYERS:
        tol[f"l{layer}"] = ceil_1sig(1.5 * max(floors[f"{v}_{kind}{layer}"] for v in FLOOR_VARIANTS for kind in "kv"))
        if f"{FLOOR_VARIANTS[0]}_ksr{layer}" in floors:
            tol[f"ksr{layer}"] = ceil_1sig(1.5 * max(floors[f"{v}_ksr{layer}"] for v in FLOOR_VARIANTS))
    return tol


def _to_mx(bits) -> mx.array:
    return mx.array(from_bits(bits)).astype(mx.bfloat16)


def _require(scene: str) -> None:
    if _meta(scene) is None:
        pytest.skip(f"{_dir(scene)} missing or incomplete: generate it with "
                    f"python -m tests.parity.gen_fixtures --scene {scene}")


@pytest.fixture(scope="module")
def loaded():
    """The one scene held in memory: its MLX components and the block case derived from them. Tests are grouped by
    scene (tests/conftest.py), so each scene loads once; switching scenes frees the previous one first (a 5B DiT per
    scene)."""
    state = {}
    yield state
    state.clear()
    gc.collect()
    mx.clear_cache()


@pytest.fixture
def comps(scene, loaded):
    from matrix_game_mlx.inference.loader import ModelComponents
    _require(scene)
    if loaded.get("scene") != scene:
        loaded.clear()
        gc.collect()
        mx.clear_cache()
        meta = _meta(scene)
        inputs = np.load(_dir(scene) / "inputs.npz")
        comps = ModelComponents(scene, "quality")
        assert (comps.sink_mode, comps.sink_rope_mode) == (meta["sink_mode"], meta["sink_rope_mode"])
        comps.load()
        comps.prompt = mx.array(inputs["prompt_embeds"]).astype(mx.bfloat16)     # same embeddings as the torch run
        comps.sink_latent = mx.array(inputs["sink_latent"]).astype(mx.bfloat16)
        loaded.update(scene=scene, comps=comps)
    return loaded["comps"]


def _inject_state(kv, fx) -> None:
    """Fill MLX KV caches with torch's state at the chunk boundary (all layers)."""
    for i, c in enumerate(kv):
        n = int(fx[f"state_lei{i}"])
        c["k"], c["v"] = mx.zeros_like(c["k"]), mx.zeros_like(c["v"])
        if n:                                                                  # racing chunk 0 starts empty
            c["k"][:, :n], c["v"][:, :n] = _to_mx(fx[f"state_k{i}"]), _to_mx(fx[f"state_v{i}"])
        c["k_sink_raw"] = _to_mx(fx[f"state_ksr{i}"])
        c["global_end_index"], c["local_end_index"] = int(fx[f"state_gei{i}"]), n


# Thresholds of the egypt calls as fixed in milestone 1 (max variant per entry: "sdpa" native bf16 SDPA, "bf16"
# native CPU bf16, "f64" float64-accumulated GEMM/conv/SDPA). Calls 0 / 1 are the t=1000 / t=833 denoising steps,
# call 2 the clean KV pass (t=0: pred threshold 0).
M1_EGYPT_CALL_TOL = {
    ("calls_chunk0", 0): (3e-2, 4e-4, 5e-2),  # pred sdpa 1.78e-2 / bf16 1.68e-2 / f64 1.70e-2; l0 bf16 2.16e-4; l29 sdpa 2.97e-2
    ("calls_chunk0", 1): (2e-2, 3e-4, 4e-2),  # pred sdpa 1.14e-2 / bf16 1.14e-2 / f64 1.17e-2; l0 bf16 1.75e-4; l29 f64 2.12e-2
    ("calls_chunk0", 2): (0.0, 3e-4, 3e-2),   # pred 0 / 0 / 0; l0 bf16 1.45e-4; l29 bf16 1.86e-2
    ("calls_chunk1", 0): (2e-1, 3e-4, 3e-1),  # pred sdpa 5.22e-2 / bf16 8.54e-2 / f64 4.92e-2; l0 bf16 1.75e-4; l29 bf16 1.57e-1
    ("calls_chunk1", 1): (5e-2, 3e-4, 1e-1),  # pred sdpa 2.43e-2 / bf16 3.00e-2 / f64 3.03e-2; l0 f64 1.49e-4; l29 f64 6.46e-2
    ("calls_chunk1", 2): (0.0, 3e-4, 3e-2),   # pred 0 / 0 / 0; l0 f64 1.54e-4; l29 f64 1.89e-2
    ("calls_chunk2", 0): (2e-1, 4e-4, 3e-1),  # pred sdpa 5.87e-2 / bf16 6.90e-2 / f64 6.77e-2; l0 f64 2.04e-4; l29 bf16 1.46e-1
    ("calls_chunk2", 1): (9e-2, 3e-4, 2e-1),  # pred sdpa 5.16e-2 / bf16 5.52e-2 / f64 4.94e-2; l0 bf16 1.34e-4; l29 bf16 1.21e-1
    ("calls_chunk2", 2): (0.0, 3e-4, 4e-2),   # pred 0 / 0 / 0; l0 bf16 1.35e-4; l29 bf16 2.54e-2
    ("calls_long22", 0): (7e-2, 3e-4, 2e-1),  # pred sdpa 3.01e-2 / bf16 2.79e-2 / f64 4.11e-2; l0 f64 1.78e-4; l29 f64 8.08e-2
    ("calls_long22", 1): (4e-2, 3e-4, 8e-2),  # pred sdpa 2.29e-2 / bf16 2.06e-2 / f64 2.18e-2; l0 bf16 1.97e-4; l29 sdpa 4.72e-2
    ("calls_long22", 2): (0.0, 2e-4, 3e-2),   # pred 0 / 0 / 0; l0 f64 1.20e-4; l29 sdpa 1.94e-2
}


def test_call_tol_rule_reproduces_m1_egypt_table():
    """call_tol on egypt's meta.json floors gives exactly the milestone-1 table (pins the rounding rule)."""
    if _meta("ancient-egypt") is None:
        pytest.skip(f"{_dir('ancient-egypt')} missing: generate it with python -m tests.parity.gen_fixtures")
    derived = {}
    for name, rows in _meta("ancient-egypt")["call_floors"]["calls"].items():
        with np.load(_dir("ancient-egypt") / f"{name}.npz") as fx:
            for j, row in enumerate(rows):
                tol = call_tol(row, t_zero=not fx[f"c{j}_t"].any())
                derived[(name, j)] = (tol["pred"], *(tol[f"l{layer}"] for layer in KV_LAYERS))
    assert derived == M1_EGYPT_CALL_TOL


@needs_weights
@pytest.mark.parametrize("scene", SCENES)
def test_sink_encoding_production_path(scene, comps):
    """The pack's sink image through ModelComponents' encode path (scene sink mode) vs the torch-encoded sink
    (rel L2 < 2e-2)."""
    from matrix_game_mlx.inference.stream import encode_sink_image
    torch_sink = np.load(_dir(scene) / "inputs.npz")["sink_latent"]
    assert comps.sink_image == comps.pack / comps.cfg["scenes"][scene]["sink_image"]
    err = rel_l2(encode_sink_image(comps.vae, comps.sink_image, "wan", first_frame=comps.sink_mode == "first_frame"),
                 torch_sink)
    print(f"{scene} sink rel L2 {err:.3e}")
    assert err < 2e-2, err


@pytest.fixture
def block_case(scene, comps, loaded):
    if "block_case" not in loaded:
        d, meta = _dir(scene), _meta(scene)
        name, j = meta.get("block_call", ["calls_chunk0", 0])                # absent in the milestone-1 egypt meta
        gates = [b["gate"] for b in meta["block_floors"]["blocks"]]
        loaded["block_case"] = (np.load(d / f"{name}.npz"), j,
                                np.load(d / f"blocks_{name.removeprefix('calls_')}_call{j}.npz"), gates)
    return loaded["block_case"]


@needs_weights
@pytest.mark.parametrize("i", range(30))
@pytest.mark.parametrize("scene", SCENES)
def test_block_parity(scene, comps, block_case, i):
    """Block i on torch's exact input, e0, context and sink KV vs torch's output; gate 1.5 x max torch variant."""
    from matrix_game_mlx.inference.wrappers import init_crossattn_cache
    fx, j, blocks, gates = block_case
    stream = comps.stream(seed=0)
    _inject_state(stream.kv, fx)
    xc = init_crossattn_cache(stream.gen, 1, mx.bfloat16)                     # cross-attn K/V from torch's context
    model = stream.gen.model
    y = model.blocks[i](_to_mx(blocks[f"x{i}"]), _to_mx(blocks["e0"]), mx.array([405]), [[1, 15, 27]], model.freqs,
                        _to_mx(blocks["context"]), None, None, kv_cache=stream.kv[i], crossattn_cache=xc[i],
                        current_start=int(fx[f"c{j}_start"]), cache_start=0, sink_rope_mode=comps.sink_rope_mode)
    err = rel_l2(y, from_bits(blocks[f"x{i + 1}"]))
    print(f"{scene} block {i}: rel L2 {err:.3e} gate {gates[i]:.3e} ratio {err / gates[i]:.2f}")
    assert err <= gates[i], (err, gates[i])


@needs_weights
@pytest.mark.parametrize("scene,name,j", _call_cases())
def test_call_parity(scene, comps, name, j):
    path = _dir(scene) / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"{path.name} missing: generate it with gen_fixtures --scene {scene} --long (then --calls-only)")
    fx = np.load(path)
    tol = call_tol(_meta(scene)["call_floors"]["calls"][name][j], t_zero=not fx[f"c{j}_t"].any())
    stream = comps.stream(seed=0)
    _inject_state(stream.kv, fx)                                               # torch's state at the chunk boundary
    _, pred = stream.gen(noisy_image_or_video=_to_mx(fx[f"c{j}_x"]),
                         conditional_dict={"prompt_embeds": comps.prompt, "control_latent": _to_mx(fx["control"])},
                         timestep=mx.array(fx[f"c{j}_t"]), kv_cache=stream.kv, crossattn_cache=stream.xc,
                         current_start=int(fx[f"c{j}_start"]), drop_control=bool(fx[f"c{j}_drop"]),
                         sink_rope_mode=comps.sink_rope_mode)
    m = {"pred": rel_l2(pred, from_bits(fx[f"c{j}_pred"]))}
    for layer in KV_LAYERS:
        for kind in ("k", "v"):
            want = from_bits(fx[f"c{j}_{kind}{layer}"])
            got = to_np(stream.kv[layer][kind][:, :stream.kv[layer]["local_end_index"]])
            assert got.shape == want.shape, f"{kind}{layer}: cache end index differs from torch"
            np.testing.assert_array_equal(got[:, :-405], want[:, :-405], err_msg=f"{kind}{layer} history")
            m[f"{kind}{layer}"] = rel_l2(written_slot(got), written_slot(want))
    # Torch's post-call end indices and k_sink_raw follow upstream CausalWanSelfAttention (egypt's derivation was
    # verified against a torch replay of all 12 calls): global_end_index = current_start + new tokens, and the raw k
    # of the written slot goes to k_sink_raw iff the slot starts inside the sink (sink_rows_written). Only racing's
    # chunk-0 clean pass (slot 0 of an empty cache) does that; torch's raw k is recorded for layers 0/29
    # (c{j}_ksr{layer}) and gated there, the other layers' written rows have no torch reference.
    tokens = fx[f"c{j}_x"].shape[1] * stream.fsl
    lei = fx[f"c{j}_k{KV_LAYERS[0]}"].shape[1]
    gei = int(fx[f"c{j}_start"]) + tokens
    rows = sink_rows_written(lei, tokens, stream.fsl)
    keep = np.ones(stream.fsl, dtype=bool)                                     # k_sink_raw rows the call leaves alone
    keep[rows.start:rows.stop] = False
    for i, c in enumerate(stream.kv):
        assert (c["global_end_index"], c["local_end_index"]) == (gei, lei), f"layer {i} end indices"
        np.testing.assert_array_equal(to_np(c["k_sink_raw"])[:, keep], from_bits(fx[f"state_ksr{i}"])[:, keep],
                                      err_msg=f"layer {i} k_sink_raw")
    if rows:
        for layer in KV_LAYERS:
            assert f"c{j}_ksr{layer}" in fx.files and f"ksr{layer}" in tol, \
                f"call writes into the sink but {path.name} has no ksr{layer}: rerun gen_fixtures --scene {scene} --calls-only"
            m[f"ksr{layer}"] = rel_l2(stream.kv[layer]["k_sink_raw"], from_bits(fx[f"c{j}_ksr{layer}"]))
    print(f"{scene} {name} call {j}: " + ", ".join(f"{k}={v:.3g}" for k, v in m.items()) + f" | tol {tol}")
    assert m["pred"] <= tol["pred"], (m, tol)
    for layer in KV_LAYERS:
        assert max(m[f"k{layer}"], m[f"v{layer}"]) <= tol[f"l{layer}"], (m, tol)
        if rows:
            assert m[f"ksr{layer}"] <= tol[f"ksr{layer}"], (m, tol)


class _StepRecorder:
    """Record pred_x0 of every control-conditioned generator call (pred_step0 is the first one per chunk)."""

    def __init__(self, gen):
        self.gen, self.preds = gen, []

    def __call__(self, **kw):
        out = self.gen(**kw)
        if not kw.get("drop_control"):
            self.preds.append(to_np(out[1]))
        return out


def _run_stream(scene, comps):
    """The full MLX stream on the fixture inputs, per chunk: control, pred_step0 (None without denoising), pred,
    decoded pixels and layer 0/29 K caches."""
    inputs = np.load(_dir(scene) / "inputs.npz")
    stream = comps.stream(seed=0)
    recorder = _StepRecorder(stream.gen)
    stream.gen = recorder
    chunks, frame = [], 0
    for i in range(_meta(scene)["chunks"]):
        px = mx.array(inputs[f"pixels_{i}"])
        recorder.preds.clear()
        c = stream.push(px, frame_start=frame, initial_noise=mx.array(inputs[f"initial_noise_{i}"]),
                        renoise=[mx.array(inputs[f"renoise_{i}"])])
        frame += px.shape[0]
        chunks.append(dict(control=to_np(c.control), n_denoising=len(recorder.preds),
                           pred_step0=recorder.preds[0] if recorder.preds else None, pred=to_np(c.latent),
                           decoded=to_np(c.pixels), kv_k_layer0=to_np(stream.kv[0]["k"]),
                           kv_k_layer29=to_np(stream.kv[29]["k"])))
    return chunks


@needs_weights
@pytest.mark.parametrize("scene", SCENES)
def test_stream_end_to_end(scene, comps):
    """Chunk 0 (and racing chunk 1) gated with the spec thresholds; other chunks reported (see module docstring)."""
    d, chunks = _dir(scene), _run_stream(scene, comps)
    first_frame = comps.sink_mode == "first_frame"
    gated = {0, 1} if first_frame else {0}
    for i, c in enumerate(chunks):
        ref = np.load(d / f"chunk_{i}.npz")
        keys = ("control", "pred_step0", "pred", "kv_k_layer0", "kv_k_layer29")
        if first_frame and i == 0:
            # No denoising call on either side: pred is the sink latent itself
            assert c["n_denoising"] == 0 and "pred_step0" not in ref.files, (c["n_denoising"], ref.files)
            np.testing.assert_array_equal(c["pred"], to_np(comps.sink_latent), err_msg="chunk 0 pred vs sink latent")
            np.testing.assert_array_equal(c["pred"], ref["pred"], err_msg="chunk 0 pred vs torch")
            keys = tuple(k for k in keys if k != "pred_step0")
        m = {k: rel_l2(c[k], ref[k]) for k in keys}
        m["decoded_psnr"] = psnr(c["decoded"], ref["decoded"])
        print(f"{scene} end-to-end chunk {i}{'' if i in gated else ' (info)'}: "
              + ", ".join(f"{k}={v:.4g}" for k, v in m.items()))
        if i in gated:
            assert m["control"] < 2e-2, m
            assert m["pred"] < 2e-2, m
            assert m["kv_k_layer0"] < 2e-2 and m["kv_k_layer29"] < 2e-2, m
            assert m["decoded_psnr"] > 35.0, m
