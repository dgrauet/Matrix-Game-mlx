"""tests/compare_runtimes.py helpers: the input pan and the comparison of two run dumps (no model)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tests.compare_runtimes import Pan, compare


def test_pan_is_the_fixture_pan(tmp_path):
    from tests.parity.gen_fixtures import pan_frames
    rng = np.random.default_rng(0)
    path = tmp_path / "sink.png"
    Image.fromarray(rng.integers(0, 256, (720, 1000, 3), dtype=np.uint8)).save(path)
    want = pan_frames(SimpleNamespace(sink_image=path), 40)       # 136 px of travel: wraps (ping-pong) at frame 17
    pan = Pan(path)
    assert np.array_equal(np.stack([pan.frame(i) for i in range(40)]), want)
    bumped = pan.frame(3, perturb=True)
    assert np.array_equal(bumped, np.minimum(want[3] + np.float32(2 / 255), 1)) and bumped.max() <= 1


def _dump(path, runtime, preds, decoded, sink_offset=1):
    path.mkdir()
    np.savez(path / "chunks.npz", **{f"pred_{i}": p for i, p in enumerate(preds)},
             **{f"decoded_{i}": d.astype(np.float16) for i, d in enumerate(decoded)})
    meta = {"scene": "castle", "preset": "realtime", "chunks": len(preds), "seed": 0, "sink_mode": "kv_only",
            "steps": [1000], "input": "pan", "dit_runtime": runtime, "window": 21, "sink_offset": sink_offset,
            "per_chunk": [{"index": i, "position": i + sink_offset} for i in range(len(preds))]}
    (path / "meta.json").write_text(json.dumps(meta))


def test_compare_reports_per_chunk_and_window_summaries(tmp_path):
    rng = np.random.default_rng(1)
    preds = [rng.standard_normal((1, 1, 48, 30, 54)).astype(np.float32) for _ in range(4)]
    decoded = [rng.uniform(-1, 1, (4, 3, 8, 8)).astype(np.float32) for _ in range(4)]
    other = [p.copy() for p in preds]
    other[3] = other[3] * 1.01
    _dump(tmp_path / "a", "native", preds, decoded)
    _dump(tmp_path / "b", "ring-fused", other, decoded)
    result = compare(tmp_path / "a", tmp_path / "b", window_chunk=2, json_path=tmp_path / "c.json")
    assert result["first_non_bitwise_chunk"] == 3 and [r["pred_bitwise"] for r in result["per_chunk"]] == [
        True, True, True, False]
    assert result["per_chunk"][3]["pred_rel_l2"] == pytest.approx(0.01, rel=1e-6)
    assert result["before_window"]["count"] == 2 and result["before_window"]["pred_rel_l2_max"] == 0
    assert result["from_window"]["chunks"] == [2, 3] and result["from_window"]["pred_rel_l2_max"] == pytest.approx(0.01)
    assert result["from_window"]["decoded_psnr_min"] == float("inf")
    assert json.loads((tmp_path / "c.json").read_text())["other"] == "ring-fused"


def test_compare_refuses_runs_with_other_inputs(tmp_path):
    preds = [np.zeros((1, 1, 48, 30, 54), np.float32)]
    decoded = [np.zeros((1, 3, 8, 8), np.float32)]
    _dump(tmp_path / "a", "native", preds, decoded)
    _dump(tmp_path / "b", "ring-fused", preds, decoded)
    meta = json.loads((tmp_path / "b" / "meta.json").read_text())
    meta["seed"] = 1
    (tmp_path / "b" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="seed"):
        compare(tmp_path / "a", tmp_path / "b", window_chunk=21)


@pytest.mark.parametrize("mode", ["ring-fused", "apple", "apple-fast"])
def test_run_meta_records_dtype_and_window(mode):
    """A run's meta names the DiT dtype and the attention window the stream set (tiny DiT, stand-in codec)."""
    import mlx.core as mx
    from matrix_game_mlx.inference.runtime import RUNTIME_SPECS
    from matrix_game_mlx.inference.stream import CastleStream
    from tests.compare_runtimes import dit_meta
    from tests.test_runtime import _PoolVAE, _stream_model
    stream = CastleStream(_stream_model(mode=mode), _PoolVAE(), mx.zeros((1, 8, 32), mx.bfloat16),
                          vae_backend="taehv", steps=(1000,), first_steps=(1000,))
    want = {"float16": mx.float16, "bfloat16": mx.bfloat16}
    meta = dit_meta(stream)
    assert set(meta) == {"dit_dtype", "window"} and meta["window"] == RUNTIME_SPECS[mode].window
    assert want[meta["dit_dtype"]] == RUNTIME_SPECS[mode].dtype
    stream.close()


def test_compare_reports_dtype_and_window(tmp_path, capsys):
    """Runs that differ only in the runtime (dtype, window) compare; the result and the header name both."""
    preds = [np.ones((1, 1, 48, 30, 54), np.float32)]
    decoded = [np.zeros((1, 3, 8, 8), np.float32)]
    _dump(tmp_path / "a", "native", preds, decoded)
    _dump(tmp_path / "b", "apple-fast", preds, decoded)
    for name, dtype, window in (("a", "bfloat16", 21), ("b", "float16", 11)):
        meta = json.loads((tmp_path / name / "meta.json").read_text())
        meta.update(dit_dtype=dtype, window=window)
        (tmp_path / name / "meta.json").write_text(json.dumps(meta))
    result = compare(tmp_path / "a", tmp_path / "b", window_chunk=21)
    assert result["dit_dtype"] == {"reference": "bfloat16", "other": "float16"}
    assert result["window"] == {"reference": 21, "other": 11}
    header = capsys.readouterr().out.splitlines()[0]
    assert "native bfloat16 window 21" in header and "apple-fast float16 window 11" in header
