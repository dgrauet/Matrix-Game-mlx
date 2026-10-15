"""Real-weight parity of the MLX stream against upstream PyTorch fixtures (tests/parity/gen_fixtures.py).

Gates are relative to a torch-vs-torch floor, not fixed thresholds. The autoregressive stream writes each chunk's
prediction into the KV cache that conditions the next chunk, so it amplifies bf16 rounding: two torch runs that
differ only in the SDPA kernel are already ~2 % apart at the first DiT pass and ~30 % apart in pred by chunk 2.
Fixed thresholds (2e-2, 35 dB) sit below that floor. `floor_chunk_{i}.npz` holds that torch variant (and, for
control, the torch VAE encoder on its other device); MLX must stay within 2x of the floor's distance to the
fixture (3 dB for decoded frames). One absolute single-forward gate remains on chunk 0's first DiT pass.
The sink KV slot is not gated separately: the SDPA variant cannot move layer-0 keys (computed before any
attention), so its floor there is 0; the sink tokens stay covered by the whole-cache kv_k_layer0/29 gates.
"""
import numpy as np
import pytest
import mlx.core as mx

from tests.conftest import fixtures_dir, psnr, rel_l2, to_np

pytestmark = [pytest.mark.needs_weights, pytest.mark.needs_fixtures]

FLOOR_FACTOR, FLOOR_MIN, PSNR_MARGIN_DB, STEP0_MAX = 2.0, 1e-3, 3.0, 5e-2


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
    floor = np.load(d / f"floor_chunk_{i}.npz")
    c = chunks[i]
    failures = []
    for k in ("control", "pred", "kv_k_layer0", "kv_k_layer29"):
        value, base = rel_l2(c[k], ref[k]), rel_l2(floor[k], ref[k])
        gate = FLOOR_FACTOR * max(base, FLOOR_MIN)
        line = f"chunk {i} {k}: mlx={value:.4g} floor={base:.4g} ratio={value / max(base, FLOOR_MIN):.2f} gate={gate:.4g}"
        print(line)
        if value > gate:
            failures.append(line)
    value, base = psnr(c["decoded"], ref["decoded"]), psnr(floor["decoded"], ref["decoded"])
    line = f"chunk {i} decoded PSNR: mlx={value:.2f} dB floor={base:.2f} dB delta={value - base:+.2f} dB"
    print(line)
    if value < base - PSNR_MARGIN_DB:
        failures.append(line)
    if i == 0:
        value = rel_l2(c["pred_step0"], ref["pred_step0"])
        line = f"chunk 0 pred_step0 (single forward): mlx={value:.4g} gate={STEP0_MAX}"
        print(line)
        if value >= STEP0_MAX:
            failures.append(line)
    assert not failures, "\n".join(failures)
