"""Real-weight parity of the MLX caption encoder against upstream umT5 (tests/parity/gen_fixtures.py --captions-only).

The MLX CaptionEncoder (resident T5 of the dynamic-prompt path) on the pack's shared T5 encodes the three scene
prompts and prompt B; each caption's rel L2 against the upstream bf16 CPU reference ($MG4_FIXTURES_DIR/captions/
embeds.npz ref_<key>) must be below ceil_1sig(1.5 x that caption's torch floor) (meta.json floors[key].max: max rel L2
of the fp32 and float64-accumulate torch variants vs the reference). Padding rows must be zero on both sides. Skipped
with the gen_fixtures command while the captions fixture is missing.
"""
import gc
import json

import numpy as np
import pytest
import mlx.core as mx

from tests.conftest import rel_l2, to_np
from tests.parity.gen_fixtures import caption_texts, captions_dir, generation_command
from tests.parity.test_real_weights import ceil_1sig

pytestmark = [pytest.mark.needs_fixtures, pytest.mark.needs_weights]

CAPTIONS = tuple(caption_texts())             # ancient-egypt, castle, racing, prompt_b


def _require() -> dict:
    path = captions_dir() / "meta.json"
    meta = json.loads(path.read_text()) if path.is_file() else None
    if meta is None or "floors" not in meta or not (captions_dir() / "embeds.npz").is_file():
        pytest.skip(f"{captions_dir()} missing or incomplete (needs embeds.npz, meta.json floors): generate it with "
                    f"{generation_command(captions=True)}")
    return meta


@pytest.fixture(scope="module")
def encoder():
    """MLX CaptionEncoder on the pack's shared T5 (loaded once, after the fixture is found)."""
    from matrix_game_mlx.inference.loader import pack_dir, resolve_shared_files
    from matrix_game_mlx.inference.text import CaptionEncoder
    from matrix_game_mlx.inference.wrappers import WanTextEncoder
    _require()
    shared = resolve_shared_files(pack_dir(()), ("t5_encoder",))
    text = WanTextEncoder(str(shared["t5_encoder.safetensors"]), str(shared["google/umt5-xxl"]))
    enc = CaptionEncoder(text, graph=True)
    yield enc
    del enc, text
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("key", CAPTIONS)
def test_caption_embeds_match_upstream(key, encoder):
    meta = _require()
    text = caption_texts()[key]
    assert meta["captions"][key] == text, (meta["captions"][key], text)
    ref = np.load(captions_dir() / "embeds.npz")[f"ref_{key}"]
    got = to_np(encoder(text))
    assert got.shape == ref.shape == (1, 512, 4096)
    n = int((np.abs(ref[0]).max(axis=-1) > 0).sum())
    assert np.all(got[0, n:] == 0) and np.all(ref[0, n:] == 0), key
    floor = meta["floors"][key]["max"]
    tol = ceil_1sig(1.5 * floor)
    err = rel_l2(got, ref)
    print(f"caption {key}: tokens={n} rel L2 {err:.3e} tol {tol:.0e} (torch floor {floor:.3e}: "
          + ", ".join(f"{k} {x:.3e}" for k, x in meta["floors"][key].items() if k != "max") + ")")
    assert err < tol, (key, err, tol)
