import hashlib
import json
import types

import pytest

from matrix_game_mlx.inference import loader


def _fake_pack(tmp_path, content=b"abc"):
    shared = tmp_path / "shared"; (shared / "google/umt5-xxl").mkdir(parents=True)
    files = {"t5_encoder.safetensors": content, "vae.safetensors": b"v",
             "google/umt5-xxl/tokenizer.json": b"{}"}
    for rel, data in files.items():
        (shared / rel).write_bytes(data)
    entries = [{"path": rel, "sha256": loader.sha256_file(shared / rel)} for rel in files]
    pack = tmp_path / "pack"; pack.mkdir()
    (pack / "split_model.json").write_text(json.dumps({"shared": {
        "repo_id": "x/y", "revision": "r", "components": {"t5_encoder": "t5_encoder.safetensors", "vae": "vae.safetensors"},
        "tokenizer": "google/umt5-xxl", "files": entries}}))
    return pack, shared


def test_local_shared_dir_ok(tmp_path, monkeypatch):
    pack, shared = _fake_pack(tmp_path)
    monkeypatch.setenv("MATRIX_GAME_4_MLX_SHARED_DIR", str(shared))
    out = loader.resolve_shared_files(pack)
    assert out["t5_encoder.safetensors"] == shared / "t5_encoder.safetensors"
    assert out["google/umt5-xxl"] == shared / "google/umt5-xxl"


def test_sha_mismatch_raises(tmp_path, monkeypatch):
    pack, shared = _fake_pack(tmp_path)
    (shared / "vae.safetensors").write_bytes(b"tampered")
    monkeypatch.setenv("MATRIX_GAME_4_MLX_SHARED_DIR", str(shared))
    with pytest.raises(ValueError, match="sha256"):
        loader.resolve_shared_files(pack)


def test_taehv_resolution_skips_wan_vae(tmp_path, monkeypatch):
    pack, shared = _fake_pack(tmp_path)
    (shared / "vae.safetensors").unlink()
    monkeypatch.setenv("MATRIX_GAME_4_MLX_SHARED_DIR", str(shared))
    out = loader.resolve_shared_files(pack, ("t5_encoder",))
    assert set(out) == {"t5_encoder.safetensors", "google/umt5-xxl"}
    with pytest.raises(FileNotFoundError):
        loader.resolve_shared_files(pack)
    # Resolved files are still sha-checked
    (shared / "t5_encoder.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        loader.resolve_shared_files(pack, ("t5_encoder",))


def test_hub_download_called_with_pinned_revision(tmp_path, monkeypatch):
    pack, shared = _fake_pack(tmp_path)
    monkeypatch.delenv("MATRIX_GAME_4_MLX_SHARED_DIR", raising=False)
    calls = []
    def fake_download(repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        return str(shared / filename)
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)
    loader.resolve_shared_files(pack)
    assert {c[2] for c in calls} == {"r"} and {c[0] for c in calls} == {"x/y"} and len(calls) == 3


@pytest.mark.parametrize("components,weights", [(None, ["*.safetensors"]), ((), []),
                                                  (("dit_castle", "taehv"), ["dit_castle.safetensors", "taehv.safetensors"])])
def test_pack_dir_from_hub_pinned(tmp_path, monkeypatch, components, weights):
    monkeypatch.delenv("MATRIX_GAME_4_MLX_WEIGHTS_DIR", raising=False)
    calls = []
    def fake_snapshot(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(tmp_path)
    monkeypatch.setattr(loader, "snapshot_download", fake_snapshot)
    assert loader.pack_dir(components) == tmp_path
    assert calls == [(loader.PACK_REPO_ID, loader.PACK_REVISION, ["*.json", "*.png", *weights])]


def _scene_pack(tmp_path, monkeypatch):
    """Fake pack with every scene; WEIGHTS_DIR points at it so model_args resolves pack sink images."""
    pack = tmp_path / "scenepack"
    (pack / "s").mkdir(parents=True)
    (pack / "s/sink.png").write_bytes(b"png-bytes")
    (pack / "s/inference.json").write_text(json.dumps({"scheduler_shift": 5.0}))
    (pack / "config.json").write_text(json.dumps({"scenes": {
        name: {"component": f"dit_{name}", "inference": "s/inference.json", "sink_image": "s/sink.png"}
        for name in ("castle", "ancient-egypt", "racing")}}))
    monkeypatch.setenv("MATRIX_GAME_4_MLX_WEIGHTS_DIR", str(pack))
    return pack


UPSTREAM_KEYS = {"scene", "scene_name", "checkpoint", "ckpt_keys", "vae_backend", "steps", "first_steps",
                 "sink_rope_mode", "sink_mode", "sink_encoder", "sink_source", "sink_image", "sink_sha256", "prompt",
                 "hq_first_frame", "prompt_mode", "prompt_interval_chunks", "t5_runtime", "dit_parallel", "dit_runtime",
                 "attention_backend", "vae_graph", "early_publish", "profile_chunks", "reference_pipeline",
                 "experimental_runtime"}


def test_config_keys_match_upstream(tmp_path, monkeypatch):
    from matrix_game_mlx.scenes import model_args
    pack = _scene_pack(tmp_path, monkeypatch)
    cfg = loader.ModelComponents(model_args("racing", "quality"), pack=pack).config()
    assert set(cfg) == UPSTREAM_KEYS | {"preset", "component"}
    assert cfg["dit_runtime"] == "native" and cfg["scene_name"] == "City Racing"
    assert cfg["hq_first_frame"] is True and cfg["sink_source"] == "image"
    assert cfg["checkpoint"] == "dit_racing.safetensors" and cfg["ckpt_keys"] == ["generator_ema"]
    assert cfg["attention_backend"] == "mlx-sdpa" and cfg["t5_runtime"] == "graph" and cfg["reference_pipeline"] == "stream"
    assert cfg["dit_parallel"] == "single" and not any(cfg[k] for k in ("vae_graph", "early_publish", "profile_chunks", "experimental_runtime"))
    assert loader.ModelComponents(model_args("racing", "quality", ["--t5-runtime", "native"]), pack=pack).config()["t5_runtime"] == "native"


def test_no_sink_config(tmp_path, monkeypatch):
    from matrix_game_mlx.scenes import model_args
    pack = _scene_pack(tmp_path, monkeypatch)
    cfg = loader.ModelComponents(model_args("racing", "quality", ["--no-sink-image"]), pack=pack).config()
    assert cfg["sink_source"] == "generated_first_frame" and cfg["sink_image"] is None and cfg["sink_sha256"] is None
    assert cfg["hq_first_frame"] is False


def test_codec_follows_vae_backend(tmp_path, monkeypatch):
    from matrix_game_mlx.scenes import model_args
    pack = _scene_pack(tmp_path, monkeypatch)
    calls = []
    def fake_pack_dir(components=None):
        calls.append(components)
        return pack
    monkeypatch.setattr(loader, "pack_dir", fake_pack_dir)
    class Stop(Exception):
        pass
    def stop(*args):
        raise Stop
    shared = []
    monkeypatch.setattr(loader, "resolve_shared_files", lambda p, c=None: shared.append(c) or stop())
    args = model_args("castle", "quality", ["--vae-backend", "taehv"])
    calls.clear()  # model_args itself reads the pack config once (sink image path)
    m = loader.ModelComponents(args)
    assert m.args.vae_backend == "taehv" and calls == [()]
    with pytest.raises(Stop):
        m.load()
    assert calls == [(), ("dit_castle", "taehv")] and shared == [("t5_encoder",)]
    shared.clear()
    args = model_args("castle", "realtime", ["--vae-backend", "wan"])
    calls.clear()
    m = loader.ModelComponents(args)
    with pytest.raises(Stop):
        m.load()
    assert calls == [(), ("dit_castle",)] and shared == [("t5_encoder", "vae")]


@pytest.mark.parametrize("mode,scene", [("dynamic", "racing"), ("fixed", "castle")])
def test_dynamic_keeps_caption_encoder(tmp_path, monkeypatch, mode, scene):
    import mlx.core as mx
    from matrix_game_mlx.inference import loading, stream, text, wrappers
    from matrix_game_mlx.scenes import model_args
    pack = _scene_pack(tmp_path, monkeypatch)
    embeds = mx.ones((1, 512, 4096), dtype=mx.bfloat16)
    class FakeText:
        def __init__(self, *a):
            pass
        def __call__(self, prompts):
            return {"prompt_embeds": embeds}
    seen = {}
    class FakeCaption:
        def __init__(self, t, *, graph=False, capacity=8):
            seen["graph"] = graph
        def __call__(self, prompt):
            seen["prompt"] = prompt
            return embeds * 2
    monkeypatch.setattr(wrappers, "WanTextEncoder", FakeText)
    monkeypatch.setattr(text, "CaptionEncoder", FakeCaption)
    monkeypatch.setattr(loading, "build_dit", lambda *a, **k: types.SimpleNamespace(model=types.SimpleNamespace()))
    monkeypatch.setattr(wrappers, "WanVAE22Wrapper", lambda *a, **k: object())
    monkeypatch.setattr(stream, "encode_sink_image", lambda *a, **k: mx.zeros((1, 48, 1, 4, 4)))
    shared = tmp_path / "t5"
    shared.write_bytes(b"")
    monkeypatch.setattr(loader, "resolve_shared_files", lambda p, c=None: {
        "t5_encoder.safetensors": shared, "google/umt5-xxl": shared, "vae.safetensors": shared})
    args = model_args(scene, "quality", ["--prompt-mode", mode, "--t5-runtime", "native"])
    m = loader.ModelComponents(args, pack=pack)
    m.load()
    assert m.sink_sha256 == hashlib.sha256(b"png-bytes").hexdigest()
    if mode == "dynamic":
        assert isinstance(m.caption_encoder, FakeCaption) and seen == {"graph": False, "prompt": args.prompt}
        assert mx.array_equal(m.prompt, embeds * 2)
    else:
        assert m.caption_encoder is None and mx.array_equal(m.prompt, embeds)


@pytest.mark.parametrize("preset,argv,runtime", [("realtime", [], "ring-fused"), ("quality", [], "native"),
                                                ("realtime", ["--dit-runtime", "native"], "native"),
                                                ("quality", ["--dit-runtime", "ring-fused"], "ring-fused")])
def test_load_installs_dit_runtime(tmp_path, monkeypatch, preset, argv, runtime):
    """load() installs the resolved runtime on the DiT and config() (render.json, session report) reports it."""
    import mlx.core as mx
    from matrix_game_mlx.inference import loading, stream, taehv, wrappers
    from matrix_game_mlx.inference.runtime import CastleRuntime
    from matrix_game_mlx.scenes import model_args
    pack = _scene_pack(tmp_path, monkeypatch)
    embeds = mx.ones((1, 512, 4096), dtype=mx.bfloat16)
    monkeypatch.setattr(wrappers, "WanTextEncoder", lambda *a: (lambda prompts: {"prompt_embeds": embeds}))
    gen = types.SimpleNamespace(model=types.SimpleNamespace())
    monkeypatch.setattr(loading, "build_dit", lambda *a, **k: gen)
    monkeypatch.setattr(wrappers, "WanVAE22Wrapper", lambda *a, **k: object())
    monkeypatch.setattr(taehv.TAEHV, "from_pack", staticmethod(lambda *a, **k: object()))
    monkeypatch.setattr(stream, "encode_sink_image", lambda *a, **k: mx.zeros((1, 48, 1, 4, 4)))
    shared = tmp_path / "t5"
    shared.write_bytes(b"")
    monkeypatch.setattr(loader, "resolve_shared_files", lambda p, c=None: {
        "t5_encoder.safetensors": shared, "google/umt5-xxl": shared, "vae.safetensors": shared})
    m = loader.ModelComponents(model_args("castle", preset, argv), pack=pack)
    assert m.config()["dit_runtime"] == runtime
    m.load()
    installed = gen.model.castle_runtime
    assert (installed is None) if runtime == "native" else (isinstance(installed, CastleRuntime) and installed.mode == runtime)
    cfg = m.config()
    assert cfg["dit_runtime"] == runtime and cfg["experimental_runtime"] is (runtime != "native")


def test_pack_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MATRIX_GAME_4_MLX_WEIGHTS_DIR", str(tmp_path))
    assert loader.pack_dir() == tmp_path


@pytest.mark.parametrize("preset,steps,backend", [("quality", (1000, 500), "wan"), ("realtime", (1000,), "taehv")])
def test_preset_resolution(tmp_path, monkeypatch, preset, steps, backend):
    from matrix_game_mlx.scenes import model_args
    pack = _scene_pack(tmp_path, monkeypatch)
    components = loader.ModelComponents(model_args("castle", preset), pack=pack)
    cfg = components.config()
    assert components.args.steps == steps and cfg["steps"] == list(steps) and cfg["first_steps"] == list(steps)
    assert cfg["vae_backend"] == backend and cfg["preset"] == preset and cfg["component"] == "dit_castle"
