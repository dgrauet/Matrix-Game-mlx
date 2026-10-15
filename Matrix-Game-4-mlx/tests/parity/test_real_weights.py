"""Real-weight parity of the MLX stream against upstream PyTorch fixtures (tests/parity/gen_fixtures.py).

Three levels:
- Per block (isolated): for chunk 0's first denoising call, each MLX block gets torch's exact block input (and
  shared e0 / context / sink KV); its output must be within 1.5 x the largest rel L2 between torch and its own
  kernel variants on that block (meta.json "block_floors"). This is the check that nothing diverges
  systematically.
- Per call (teacher forced): torch's exact cache state at a chunk boundary and torch's exact inputs are injected,
  one MLX generator call runs, and its pred plus the layer 0/29 KV it writes are compared with torch. Each call's
  threshold is 1.5 x the largest rel L2 between torch and its kernel variants on the same call (meta.json
  "call_floors"), rounded up to one significant digit. The rest of the cache (history, sink, rolled window), the
  end indices and k_sink_raw of every layer must match torch exactly. Long chunk 22 (needs `gen_fixtures --long`,
  skipped otherwise) runs past the 21-frame window, so the rolling eviction and the fixed-sink re-RoPE run with
  real weights.
- End to end: the full MLX stream from the same inputs. The autoregressive stream amplifies bf16 rounding chunk
  over chunk (torch against itself reaches ~0.3 rel L2 by chunk 2), so only chunk 0 is gated, with the spec
  thresholds; chunks 1 and 2 are reported.
"""
import json

import numpy as np
import pytest
import mlx.core as mx

from tests.conftest import fixtures_dir, psnr, rel_l2, to_np
from tests.parity.gen_fixtures import KV_LAYERS, LONG_CALL_CHUNK, from_bits, written_slot

pytestmark = [pytest.mark.needs_weights, pytest.mark.needs_fixtures]

LONG = f"calls_long{LONG_CALL_CHUNK}"
# (fixture, call) -> (pred, layer-0 KV slot, layer-29 KV slot) rel L2 thresholds: 1.5 x the max over the three
# torch kernel variants of that call ("sdpa": native bf16 SDPA, "bf16": native CPU bf16, "f64": float64-accumulated
# GEMM/conv/SDPA; meta.json "call_floors"), rounded up to one significant digit. The comments give the max variant
# per entry. KV numbers are max(k, v) of the slot the call writes. Calls 0 / 1 are the t=1000 / t=833 denoising
# steps, call 2 the clean KV pass (t=0: pred = input exactly, so its pred threshold is 0).
CALL_TOL = {
    ("calls_chunk0", 0): (3e-2, 4e-4, 5e-2),  # pred sdpa 1.78e-2 / bf16 1.68e-2 / f64 1.70e-2; l0 bf16 2.16e-4; l29 sdpa 2.97e-2
    ("calls_chunk0", 1): (2e-2, 3e-4, 4e-2),  # pred sdpa 1.14e-2 / bf16 1.14e-2 / f64 1.17e-2; l0 bf16 1.75e-4; l29 f64 2.12e-2
    ("calls_chunk0", 2): (0.0, 3e-4, 3e-2),   # pred 0 / 0 / 0; l0 bf16 1.45e-4; l29 bf16 1.86e-2
    ("calls_chunk1", 0): (2e-1, 3e-4, 3e-1),  # pred sdpa 5.22e-2 / bf16 8.54e-2 / f64 4.92e-2; l0 bf16 1.75e-4; l29 bf16 1.57e-1
    ("calls_chunk1", 1): (5e-2, 3e-4, 1e-1),  # pred sdpa 2.43e-2 / bf16 3.00e-2 / f64 3.03e-2; l0 f64 1.49e-4; l29 f64 6.46e-2
    ("calls_chunk1", 2): (0.0, 3e-4, 3e-2),   # pred 0 / 0 / 0; l0 f64 1.54e-4; l29 f64 1.89e-2
    ("calls_chunk2", 0): (2e-1, 4e-4, 3e-1),  # pred sdpa 5.87e-2 / bf16 6.90e-2 / f64 6.77e-2; l0 f64 2.04e-4; l29 bf16 1.46e-1
    ("calls_chunk2", 1): (9e-2, 3e-4, 2e-1),  # pred sdpa 5.16e-2 / bf16 5.52e-2 / f64 4.94e-2; l0 bf16 1.34e-4; l29 bf16 1.21e-1
    ("calls_chunk2", 2): (0.0, 3e-4, 4e-2),   # pred 0 / 0 / 0; l0 bf16 1.35e-4; l29 bf16 2.54e-2
    (LONG, 0): (7e-2, 3e-4, 2e-1),            # pred sdpa 3.01e-2 / bf16 2.79e-2 / f64 4.11e-2; l0 f64 1.78e-4; l29 f64 8.08e-2
    (LONG, 1): (4e-2, 3e-4, 8e-2),            # pred sdpa 2.29e-2 / bf16 2.06e-2 / f64 2.18e-2; l0 bf16 1.97e-4; l29 sdpa 4.72e-2
    (LONG, 2): (0.0, 2e-4, 3e-2),             # pred 0 / 0 / 0; l0 f64 1.20e-4; l29 sdpa 1.94e-2
}


def _to_mx(bits) -> mx.array:
    return mx.array(from_bits(bits)).astype(mx.bfloat16)


@pytest.fixture(scope="module")
def comps():
    from matrix_game_mlx.inference.loader import ModelComponents
    inputs = np.load(fixtures_dir() / "egypt_quality" / "inputs.npz")
    comps = ModelComponents("ancient-egypt", "quality")
    comps.load()
    comps.prompt = mx.array(inputs["prompt_embeds"]).astype(mx.bfloat16)     # same embeddings as the torch run
    comps.sink_latent = mx.array(inputs["sink_latent"]).astype(mx.bfloat16)
    return comps


def _inject_state(kv, fx) -> None:
    """Fill MLX KV caches with torch's state at the chunk boundary (all layers)."""
    for i, c in enumerate(kv):
        n = int(fx[f"state_lei{i}"])
        c["k"], c["v"] = mx.zeros_like(c["k"]), mx.zeros_like(c["v"])
        c["k"][:, :n], c["v"][:, :n] = _to_mx(fx[f"state_k{i}"]), _to_mx(fx[f"state_v{i}"])
        c["k_sink_raw"] = _to_mx(fx[f"state_ksr{i}"])
        c["global_end_index"], c["local_end_index"] = int(fx[f"state_gei{i}"]), n


@pytest.fixture(scope="module")
def block_case(comps):
    d = fixtures_dir() / "egypt_quality"
    gates = [b["gate"] for b in json.loads((d / "meta.json").read_text())["block_floors"]["blocks"]]
    return np.load(d / "calls_chunk0.npz"), np.load(d / "blocks_chunk0_call0.npz"), gates


@pytest.mark.parametrize("i", range(30))
def test_block_parity(comps, block_case, i):
    """Block i on torch's exact input, e0, context and sink KV vs torch's output; gate 1.5 x max torch variant."""
    from matrix_game_mlx.inference.wrappers import init_crossattn_cache
    fx, blocks, gates = block_case
    stream = comps.stream(seed=0)
    _inject_state(stream.kv, fx)
    xc = init_crossattn_cache(stream.gen, 1, mx.bfloat16)                     # cross-attn K/V from torch's context
    model = stream.gen.model
    y = model.blocks[i](_to_mx(blocks[f"x{i}"]), _to_mx(blocks["e0"]), mx.array([405]), [[1, 15, 27]], model.freqs,
                        _to_mx(blocks["context"]), None, None, kv_cache=stream.kv[i], crossattn_cache=xc[i],
                        current_start=int(fx["c0_start"]), cache_start=0, sink_rope_mode="fixed")
    err = rel_l2(y, from_bits(blocks[f"x{i + 1}"]))
    print(f"block {i}: rel L2 {err:.3e} gate {gates[i]:.3e} ratio {err / gates[i]:.2f}")
    assert err <= gates[i], (err, gates[i])


@pytest.mark.parametrize("name,j", sorted(CALL_TOL))
def test_call_parity(comps, name, j):
    path = fixtures_dir() / "egypt_quality" / f"{name}.npz"
    if name == LONG and not path.exists():
        pytest.skip(f"{path.name} missing: generate it with gen_fixtures --long (then --calls-only)")
    fx = np.load(path)
    stream = comps.stream(seed=0)
    _inject_state(stream.kv, fx)                                               # torch's state at the chunk boundary
    _, pred = stream.gen(noisy_image_or_video=_to_mx(fx[f"c{j}_x"]),
                         conditional_dict={"prompt_embeds": comps.prompt, "control_latent": _to_mx(fx["control"])},
                         timestep=mx.array(fx[f"c{j}_t"]), kv_cache=stream.kv, crossattn_cache=stream.xc,
                         current_start=int(fx[f"c{j}_start"]), drop_control=bool(fx[f"c{j}_drop"]), sink_rope_mode="fixed")
    m = {"pred": rel_l2(pred, from_bits(fx[f"c{j}_pred"]))}
    for layer in KV_LAYERS:
        for kind in ("k", "v"):
            want = from_bits(fx[f"c{j}_{kind}{layer}"])
            got = to_np(stream.kv[layer][kind][:, :stream.kv[layer]["local_end_index"]])
            assert got.shape == want.shape, f"{kind}{layer}: cache end index differs from torch"
            np.testing.assert_array_equal(got[:, :-405], want[:, :-405], err_msg=f"{kind}{layer} history")
            m[f"{kind}{layer}"] = rel_l2(written_slot(got), written_slot(want))
    # Torch's post-call end indices and k_sink_raw are derived from the fixture (verified against a torch replay of
    # all 12 calls): upstream sets global_end_index = current_start + new tokens, and writes k_sink_raw only when the
    # written slot starts inside the sink, which torch's slot (at local_end_index - 405 >= 405) does not.
    lei = fx[f"c{j}_k{KV_LAYERS[0]}"].shape[1]
    assert lei - 405 >= 405, "torch's written slot overlaps the sink"
    gei = int(fx[f"c{j}_start"]) + fx[f"c{j}_x"].shape[1] * 405
    for i, c in enumerate(stream.kv):
        assert (c["global_end_index"], c["local_end_index"]) == (gei, lei), f"layer {i} end indices"
        np.testing.assert_array_equal(to_np(c["k_sink_raw"]), from_bits(fx[f"state_ksr{i}"]),
                                      err_msg=f"layer {i} k_sink_raw")
    tol = dict(zip(("pred", *(f"l{layer}" for layer in KV_LAYERS)), CALL_TOL[(name, j)]))
    print(f"{name} call {j}: " + ", ".join(f"{k}={v:.3g}" for k, v in m.items()) + f" | tol {tol}")
    assert m["pred"] <= tol["pred"], (m, tol)
    for layer in KV_LAYERS:
        assert max(m[f"k{layer}"], m[f"v{layer}"]) <= tol[f"l{layer}"], (m, tol)


class _StepRecorder:
    """Record pred_x0 of every control-conditioned generator call (pred_step0 is the first one per chunk)."""

    def __init__(self, gen):
        self.gen, self.preds = gen, []

    def __call__(self, **kw):
        out = self.gen(**kw)
        if not kw.get("drop_control"):
            self.preds.append(to_np(out[1]))
        return out


@pytest.fixture(scope="module")
def run(comps):
    d = fixtures_dir() / "egypt_quality"
    inputs = np.load(d / "inputs.npz")
    stream = comps.stream(seed=0)
    recorder = _StepRecorder(stream.gen)
    stream.gen = recorder
    chunks, frame = [], 0
    for i in range(3):
        px = mx.array(inputs[f"pixels_{i}"])
        recorder.preds.clear()
        c = stream.push(px, frame_start=frame, initial_noise=mx.array(inputs[f"initial_noise_{i}"]),
                        renoise=[mx.array(inputs[f"renoise_{i}"])])
        frame += px.shape[0]
        chunks.append(dict(control=to_np(c.control), pred_step0=recorder.preds[0], pred=to_np(c.latent),
                           decoded=to_np(c.pixels), kv_k_layer0=to_np(stream.kv[0]["k"]),
                           kv_k_layer29=to_np(stream.kv[29]["k"])))
    return d, chunks


def test_stream_end_to_end(run):
    """Chunk 0 gated with the spec thresholds; chunks 1-2 reported only (see module docstring)."""
    d, chunks = run
    for i, c in enumerate(chunks):
        ref = np.load(d / f"chunk_{i}.npz")
        m = {k: rel_l2(c[k], ref[k]) for k in ("control", "pred_step0", "pred", "kv_k_layer0", "kv_k_layer29")}
        m["decoded_psnr"] = psnr(c["decoded"], ref["decoded"])
        print(f"end-to-end chunk {i}{'' if i == 0 else ' (info)'}: " + ", ".join(f"{k}={v:.4g}" for k, v in m.items()))
        if i == 0:
            assert m["control"] < 2e-2, m
            assert m["pred"] < 2e-2, m
            assert m["kv_k_layer0"] < 2e-2 and m["kv_k_layer29"] < 2e-2, m
            assert m["decoded_psnr"] > 35.0, m
