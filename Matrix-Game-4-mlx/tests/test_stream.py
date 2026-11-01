import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from matrix_game_mlx.inference.stream import CastleStream, LATENT_SHAPE


def _bare_stream(index=0, frame=0, max_latents=8):
    s = CastleStream.__new__(CastleStream)
    s.closed, s.failed, s.index, s.frame, s.max_latents, s.shape = False, False, index, frame, max_latents, LATENT_SHAPE
    s.gen = None                                                   # no generator, hence no DiT runtime to clear
    return s


def test_validate_cadence_range_budget():
    s = _bare_stream()
    with pytest.raises(ValueError, match="consecutive"):
        s._validate(mx.zeros((4, 3, 480, 864)), frame_start=0)          # first chunk must be 1 frame
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        s._validate(mx.full((1, 3, 480, 864), 2.0), frame_start=0)
    assert s._validate(mx.zeros((1, 3, 480, 864)), frame_start=0) == 1
    s = _bare_stream(index=1, frame=1)
    with pytest.raises(ValueError, match="consecutive"):
        s._validate(mx.zeros((4, 3, 480, 864)), frame_start=2)          # frame_start must follow
    assert s._validate(mx.zeros((4, 3, 480, 864)), frame_start=1) == 4
    with pytest.raises(ValueError, match="budget"):
        _bare_stream(index=8, frame=29)._validate(mx.zeros((4, 3, 480, 864)), frame_start=29)
    s.failed = True
    with pytest.raises(RuntimeError, match="closed or failed"):
        s._validate(mx.zeros((4, 3, 480, 864)), frame_start=1)


def test_close_clears_codec_caches():
    from types import SimpleNamespace
    s = _bare_stream()
    s.vae_backend = "wan"
    s.kv, s.xc = [{"k": mx.zeros((1,))}], [{"is_init": True}]
    s.encoder_features, s.decoder = [mx.zeros((1,))] * 3, SimpleNamespace(feat=[mx.zeros((1,))] * 2)
    s.close()
    assert s.closed and s.kv == [] and s.xc == [] and s.encoder_features == [] and s.decoder.feat == []


def test_rejects_integer_sink_latent():
    from types import SimpleNamespace
    generator = SimpleNamespace(model=SimpleNamespace(patch_embedding=SimpleNamespace(weight=mx.zeros((1,), mx.bfloat16))))
    with pytest.raises(ValueError, match="floating-point"):
        CastleStream(generator, None, None, sink_latent=mx.zeros(LATENT_SHAPE, mx.int32))


def _taehv(dtype=mx.bfloat16):
    from matrix_game_mlx.inference.taehv import TAEHV
    mx.random.seed(0)
    return TAEHV("taew2_2.safetensors", dtype, load=False)


class _FakeGenerator:
    """Records DiT calls; returns a deterministic latent prediction."""

    def __init__(self):
        from types import SimpleNamespace
        from matrix_game_mlx.inference.scheduler import FlowMatchScheduler
        self.model = SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=mx.zeros((1,), mx.bfloat16)), blocks=[], num_heads=1, dim=4,
            num_layers=1, patch_size=(1, 2, 2), set_rope_max_len=lambda n: None)
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.calls = []

    def get_scheduler(self):
        return self.scheduler

    def __call__(self, **kw):
        self.calls.append(kw)
        pred = mx.random.normal(LATENT_SHAPE, key=mx.random.key(len(self.calls))) * 0.1
        return None, pred.astype(mx.bfloat16)


def test_taehv_stream_three_pushes():
    from matrix_game_mlx.inference.stream import CastleStream
    vae, ref = _taehv(), _taehv()
    ref.core.update(vae.core.parameters())                       # identical random weights
    gen = _FakeGenerator()
    stream = CastleStream(gen, vae, mx.zeros((1, 2, 4)), vae_backend="taehv", steps=(1000,), first_steps=(1000,))
    rng = np.random.default_rng(0)
    frame = 0
    for index, count in enumerate((1, 4, 4)):
        pixels = mx.array(rng.uniform(-1, 1, (count, 3, 480, 864)).astype(np.float32))
        before = len(gen.calls)
        chunk = stream.push(pixels, frame_start=frame)
        assert tuple(chunk.pixels.shape) == (count, 3, 480, 864) and bool(mx.all(mx.isfinite(chunk.pixels)))
        calls = gen.calls[before:]
        assert len(calls) == 2
        denoise, clean = calls
        assert float(denoise["timestep"][0, 0]) == 1000 and "control_latent" in denoise["conditional_dict"]
        assert not denoise.get("drop_control") and clean["drop_control"] is True
        expected = ref.encode_stream(pixels.transpose(1, 0, 2, 3)[None]).transpose(0, 2, 1, 3, 4)
        assert mx.array_equal(denoise["conditional_dict"]["control_latent"], expected)
        assert mx.array_equal(chunk.control, expected)
        frame += count
    stream.close()
    assert vae._streaming_encoder is None and vae._streaming_decoder is None and vae._streaming_encode_frames == 0


def _run_pushes(stream, counts=(1, 4, 4), **push_kw):
    rng = np.random.default_rng(0)
    frame, chunks = 0, []
    for count in counts:
        pixels = mx.array(rng.uniform(-1, 1, (count, 3, 480, 864)).astype(np.float32))
        chunks.append(stream.push(pixels, frame_start=frame, **push_kw))
        frame += count
    return chunks


def _taehv_stream(gen=None, **kw):
    from matrix_game_mlx.inference.stream import CastleStream
    return CastleStream(gen or _FakeGenerator(), _taehv(), mx.zeros((1, 2, 4)), vae_backend="taehv", steps=(1000,),
                        first_steps=(1000,), **kw)


def test_on_rgb_matches_plain_push():
    from matrix_game_mlx.inference.preprocessing import tensor_to_uint8
    plain_stream, early_stream = _taehv_stream(seed=3), _taehv_stream(seed=3)
    plain = _run_pushes(plain_stream)
    got = []
    rng = np.random.default_rng(0)
    frame = 0
    for index, count in enumerate((1, 4, 4)):
        pixels = mx.array(rng.uniform(-1, 1, (count, 3, 480, 864)).astype(np.float32))
        early = early_stream.push(pixels, frame_start=frame, on_rgb=got.append)
        frame += count
        assert early.pixels is None and early.rgb is got[-1]
        assert got[-1].dtype == np.uint8 and got[-1].shape == (count, 480, 864, 3)
        assert np.array_equal(got[-1], tensor_to_uint8(plain[index].pixels))
        assert mx.array_equal(early.latent, plain[index].latent) and mx.array_equal(early.control, plain[index].control)
    assert len(got) == 3 and plain[0].rgb is None


def test_on_rgb_runs_before_clean_kv():
    gen = _FakeGenerator()
    stream = _taehv_stream(gen)
    events = []
    real_call = gen.__call__.__func__

    class Recording(type(gen)):
        def __call__(self, **kw):
            events.append("clean" if kw.get("drop_control") else "denoise")
            return real_call(self, **kw)

    gen.__class__ = Recording
    pixels = mx.array(np.random.default_rng(0).uniform(-1, 1, (1, 3, 480, 864)).astype(np.float32))
    stream.push(pixels, frame_start=0, on_rgb=lambda rgb: events.append("callback"))
    assert events == ["denoise", "callback", "clean"]


def test_profile_timing_keys():
    plain = _run_pushes(_taehv_stream(profile=True), (1,))[0]
    assert tuple(plain.timings_ms) == ("encode", "denoise", "clean_kv", "decode")
    early = _run_pushes(_taehv_stream(profile=True), (1,), on_rgb=lambda rgb: None)[0]
    assert tuple(early.timings_ms) == ("encode", "denoise", "decode", "pack_copy", "clean_kv")
    assert all(v >= 0 for c in (plain, early) for v in c.timings_ms.values())
    assert _run_pushes(_taehv_stream(), (1,))[0].timings_ms == {}


class _FakeRuntime:
    """Records the stream's runtime hooks (upstream: prime before gameplay, clear on close)."""

    def __init__(self):
        self.events = []

    def prime(self, generator, prompt, shape, kv, xc, fsl, sink_rope_mode="fixed"):
        self.events.append(("prime", generator, shape, len(kv), len(xc), fsl, sink_rope_mode))

    def clear(self):
        self.events.append(("clear",))


def test_stream_primes_runtime_and_clears_it_on_close():
    gen = _FakeGenerator()
    gen.model.castle_runtime = runtime = _FakeRuntime()
    stream = _taehv_stream(gen, sink_rope_mode="absolute")
    assert runtime.events == [("prime", gen, LATENT_SHAPE, 1, 1, stream.fsl, "absolute")] and not gen.calls
    _run_pushes(stream, (1,))
    stream.close()
    stream.close()
    assert runtime.events[1:] == [("clear",)]                    # a new stream on the same model starts clean


def test_stream_primes_taehv_graph_after_clearing():
    """As upstream: a TAEHV stream clears the codec state, then primes the VAE graph when the VAE has one."""
    vae, events = _taehv(), []
    vae.clear_cache = lambda: events.append("clear")
    vae.prime = lambda: events.append("prime")
    CastleStream(_FakeGenerator(), vae, mx.zeros((1, 2, 4)), vae_backend="taehv", steps=(1000,), first_steps=(1000,))
    assert events == ["clear", "prime"]


def test_validate_rejects_bad_pixels_and_noise():
    s = _bare_stream()
    for bad in (mx.full((1, 3, 480, 864), -1.5), mx.full((1, 3, 480, 864), float("nan")),
                mx.full((1, 3, 480, 864), float("inf")), mx.zeros((1, 3, 480, 864), mx.int32)):
        with pytest.raises(ValueError, match=r"\[-1,1\]"):
            s._validate(bad, frame_start=0)
    with pytest.raises(ValueError, match="Initial noise"):
        s._validate(mx.zeros((1, 3, 480, 864)), frame_start=0, initial_noise=mx.full(LATENT_SHAPE, float("nan")))
    with pytest.raises(ValueError, match="Initial noise"):
        s._validate(mx.zeros((1, 3, 480, 864)), frame_start=0, initial_noise=mx.zeros((1, 2, 3)))
    assert s._validate(mx.ones((1, 3, 480, 864)), frame_start=0, initial_noise=mx.zeros(LATENT_SHAPE)) == 1


def test_push_from_worker_thread():
    from concurrent.futures import ThreadPoolExecutor
    reference = _run_pushes(_taehv_stream(seed=5))
    stream = _taehv_stream(seed=5)
    with ThreadPoolExecutor(1) as pool:
        threaded = pool.submit(_run_pushes, stream).result()
    for a, b in zip(reference, threaded):
        assert mx.array_equal(a.pixels, b.pixels) and mx.array_equal(a.latent, b.latent)


def test_encode_sink_image_taehv_clears_state(tmp_path):
    from PIL import Image
    from matrix_game_mlx.inference.stream import encode_sink_image
    path = tmp_path / "sink.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (700, 1000, 3), dtype=np.uint8)).save(path)
    vae, fresh = _taehv(), _taehv()
    fresh.core.update(vae.core.parameters())
    for first_frame in (False, True):
        latent = encode_sink_image(vae, path, "taehv", first_frame=first_frame)
        assert tuple(latent.shape) == LATENT_SHAPE
        assert vae._streaming_encoder is None and vae._streaming_encode_frames == 0 and vae._streaming_decoder is None
    pixels = mx.array(np.random.default_rng(2).uniform(-1, 1, (1, 3, 1, 480, 864)).astype(np.float32))
    assert mx.array_equal(vae.encode_stream(pixels), fresh.encode_stream(pixels))


def _dirty_pair():
    """A TAEHV codec with live streaming encoder/decoder state, plus a fresh twin with identical weights."""
    vae, fresh = _taehv(), _taehv()
    fresh.core.update(vae.core.parameters())
    rng = np.random.default_rng(3)
    vae.encode_stream(mx.array(rng.uniform(-1, 1, (1, 3, 1, 480, 864)).astype(np.float32)))
    vae.decode_stream(mx.array(rng.normal(size=LATENT_SHAPE).astype(np.float32)).astype(mx.bfloat16)
                      .transpose(0, 2, 1, 3, 4))
    assert vae._streaming_encoder is not None and vae._streaming_decoder is not None and vae._streaming_encode_frames > 0
    return vae, fresh


def _assert_clean_and_equal_to_fresh(vae, fresh):
    assert vae._streaming_encoder is None and vae._streaming_decoder is None and vae._streaming_encode_frames == 0
    pixels = mx.array(np.random.default_rng(2).uniform(-1, 1, (1, 3, 1, 480, 864)).astype(np.float32))
    assert mx.array_equal(vae.encode_stream(pixels), fresh.encode_stream(pixels))


def test_encode_sink_image_taehv_clears_dirty_state(tmp_path):
    from PIL import Image
    from matrix_game_mlx.inference.stream import encode_sink_image
    path = tmp_path / "sink.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (700, 1000, 3), dtype=np.uint8)).save(path)
    vae, fresh = _dirty_pair()
    encode_sink_image(vae, path, "taehv")
    _assert_clean_and_equal_to_fresh(vae, fresh)


def test_castle_stream_init_clears_dirty_taehv_state():
    vae, fresh = _dirty_pair()
    CastleStream(_FakeGenerator(), vae, mx.zeros((1, 2, 4)), vae_backend="taehv", steps=(1000,), first_steps=(1000,))
    _assert_clean_and_equal_to_fresh(vae, fresh)


def test_unknown_backend_raises(tmp_path):
    from matrix_game_mlx.inference.stream import encode_sink_image
    with pytest.raises(ValueError, match="Expected wan or taehv VAE backend"):
        encode_sink_image(None, tmp_path / "x.png", "bogus")
    generator = SimpleNamespace(model=SimpleNamespace(patch_embedding=SimpleNamespace(weight=mx.zeros((1,), mx.bfloat16))))
    with pytest.raises(ValueError, match="Expected wan or taehv VAE backend"):
        CastleStream(generator, None, None, vae_backend="bogus")


@pytest.mark.needs_weights
def test_model_components_stream_smoke():
    import numpy as np
    from matrix_game_mlx.inference.loader import ModelComponents
    from matrix_game_mlx.scenes import model_args

    components = ModelComponents(model_args("ancient-egypt", "quality"))
    components.load()
    stream = components.stream()
    frame = 0
    for count in (1, 4):
        t0 = time.perf_counter()
        chunk = stream.push(mx.zeros((count, 3, 480, 864)), frame_start=frame)
        mx.eval(chunk.pixels)
        print(f"chunk {chunk.index}: {time.perf_counter() - t0:.1f}s, peak {mx.get_peak_memory() / 2**30:.1f} GiB")
        assert tuple(chunk.pixels.shape) == (count, 3, 480, 864)
        assert np.isfinite(np.array(chunk.pixels)).all()
        frame += count
    stream.close()


@pytest.mark.needs_weights
def test_model_components_realtime_stream_smoke():
    from matrix_game_mlx.inference.loader import ModelComponents
    from matrix_game_mlx.scenes import model_args

    components = ModelComponents(model_args("ancient-egypt", "realtime"))
    components.load()
    stream = components.stream()
    frame = 0
    for count in (1, 4):
        t0 = time.perf_counter()
        chunk = stream.push(mx.zeros((count, 3, 480, 864)), frame_start=frame)
        mx.eval(chunk.pixels)
        print(f"realtime chunk {chunk.index}: {time.perf_counter() - t0:.1f}s, peak {mx.get_peak_memory() / 2**30:.1f} GiB")
        assert tuple(chunk.pixels.shape) == (count, 3, 480, 864)
        assert chunk.denoising_steps == 1
        assert np.isfinite(np.array(chunk.pixels.astype(mx.float32))).all()
        frame += count
    stream.close()
