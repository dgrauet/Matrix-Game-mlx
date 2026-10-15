"""Real-weight parity of the MLX stream against upstream PyTorch fixtures (tests/parity/gen_fixtures.py)."""
import numpy as np
import pytest
import mlx.core as mx

from tests.conftest import fixtures_dir, psnr, rel_l2, to_np

pytestmark = [pytest.mark.needs_weights, pytest.mark.needs_fixtures]


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
def run():
    from matrix_game_mlx.inference.loader import ModelComponents
    d = fixtures_dir() / "egypt_quality"
    inputs = np.load(d / "inputs.npz")
    comps = ModelComponents("ancient-egypt", "quality")
    comps.load()
    comps.prompt = mx.array(inputs["prompt_embeds"]).astype(mx.bfloat16)     # same embeddings as the torch run
    comps.sink_latent = mx.array(inputs["sink_latent"]).astype(mx.bfloat16)
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


@pytest.mark.parametrize("i", [0, 1, 2])
def test_chunk_parity(run, i):
    d, chunks = run
    ref = np.load(d / f"chunk_{i}.npz")
    c = chunks[i]
    m = {k: rel_l2(c[k], ref[k]) for k in ("control", "pred_step0", "pred", "kv_k_layer0", "kv_k_layer29")}
    m["decoded_psnr"] = psnr(c["decoded"], ref["decoded"])
    print(f"chunk {i}: " + ", ".join(f"{k}={v:.4g}" for k, v in m.items()))
    assert m["control"] < 2e-2, m
    assert m["pred"] < 2e-2, m
    assert m["kv_k_layer0"] < 2e-2 and m["kv_k_layer29"] < 2e-2, m
    assert m["decoded_psnr"] > 35.0, m
