import argparse

import pytest

from matrix_game_mlx import scenes
from matrix_game_mlx.scenes import model_args, parse_steps

WEIGHTS_ENV = "MATRIX_GAME_4_MLX_WEIGHTS_DIR"


@pytest.fixture(autouse=True)
def fake_pack(tmp_path, monkeypatch):
    """Offline pack holding only config.json; pack_dir never hits the network."""
    import json
    pack = tmp_path / "pack"
    pack.mkdir()
    cfg = {"scenes": {s: {"component": f"dit_{s}", "inference": f"{s}.json", "sink_image": f"{s}_sink.png"}
                      for s in scenes.SCENES}}
    (pack / "config.json").write_text(json.dumps(cfg))
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    monkeypatch.setattr("matrix_game_mlx.inference.loader.pack_dir", lambda components=None: pack)
    return pack


@pytest.mark.parametrize("preset,backend,steps,runtime", [
    ("quality", "wan", (1000, 500), "native"), ("baseline", "wan", (1000, 500), "native"),
    ("realtime", "taehv", (1000,), "ring-fused")])
def test_preset_resolution_matches_upstream(preset, backend, steps, runtime):
    a = model_args("castle", preset)
    graph = preset == "realtime"                         # upstream: realtime + TAEHV defaults to the VAE graph
    assert (a.vae_backend, a.steps, a.first_steps, a.dit_runtime, a.vae_graph) == (backend, steps, steps, runtime, graph)


def test_steps_override_keeps_preset_codec():
    a = model_args("castle", "realtime", ["--steps", "1000,500"])
    assert (a.vae_backend, a.steps, a.first_steps) == ("taehv", (1000, 500), (1000, 500))
    a = model_args("castle", "realtime", ["--steps", "1000,500", "--first-steps", "1000,750,500,250"])
    assert a.first_steps == (1000, 750, 500, 250)


@pytest.mark.parametrize("value", ["500", "1000,1000", "1000,0", "1000,1200", "x"])
def test_parse_steps_rejects(value):
    with pytest.raises(argparse.ArgumentTypeError, match="Timesteps must decrease strictly from 1000"):
        parse_steps(value)


@pytest.mark.parametrize("argv", [["--preset", "rtx5090"], ["--preset", "rtx5090-sage"]])
def test_cuda_only_rejected(argv):
    with pytest.raises(ValueError, match="CUDA-only"):
        model_args("castle", "quality", argv)


@pytest.mark.parametrize("preset", ["quality", "realtime"])
@pytest.mark.parametrize("runtime", ["kernels", "cudagraph", "ring", "ring-graph", "rtx5090", "rtx5090-sage"])
def test_cuda_only_runtimes_rejected(preset, runtime):
    with pytest.raises(ValueError) as raised:
        model_args("castle", preset, ["--dit-runtime", runtime])
    assert str(raised.value) == f"--dit-runtime {runtime} is CUDA-only; the MLX port runs native or ring-fused"


@pytest.mark.parametrize("scene", list(scenes.SCENES))
@pytest.mark.parametrize("preset", ["quality", "baseline", "realtime"])
@pytest.mark.parametrize("runtime", ["native", "ring-fused"])
def test_mlx_runtimes_pass(scene, preset, runtime):
    assert model_args(scene, preset, ["--dit-runtime", runtime]).dit_runtime == runtime


@pytest.mark.parametrize("flag", ["--model-dir", "--checkpoint", "--taehv-path"])
def test_pth_flags_rejected(flag):
    with pytest.raises(ValueError, match="MLX pack"):
        model_args("castle", "quality", [flag, "x"])


def test_dynamic_rules():
    with pytest.raises(ValueError, match="Dynamic captions are currently supported for racing only"):
        model_args("castle", "quality", ["--prompt-mode", "dynamic"])
    for n in ("0", "1025"):
        with pytest.raises(ValueError, match="Prompt interval must be 1..1024 chunks"):
            model_args("racing", "quality", ["--prompt-interval-chunks", n])
    assert model_args("racing", "quality", ["--prompt-mode", "dynamic"]).prompt_mode == "dynamic"


def test_prompt_override_and_sink_policy(fake_pack):
    assert model_args("castle", "quality", ["--prompt", "a b"]).prompt == "a b"
    assert model_args("castle", "quality").prompt == scenes.SCENES["castle"]["prompt"]
    a = model_args("racing", "quality")
    assert (a.sink_mode, a.sink_rope_mode) == ("first_frame", "absolute")
    assert a.sink_image == fake_pack / "racing_sink.png"
    assert a.asset_root.parts[-2:] == ("resources", "racing")
    a = model_args("castle", "quality")
    assert (a.sink_mode, a.sink_rope_mode) == ("kv_only", "fixed")
    assert a.asset_root.parts[-2:] == ("resources", "game")
    assert model_args("castle", "quality", ["--no-sink-image"]).sink_image is None
    with pytest.raises(SystemExit):
        model_args("castle", "quality", ["--sink-image", "a.png", "--no-sink-image"])


def test_missing_resources_only_with_local_pack(tmp_path, monkeypatch, fake_pack):
    model_args("castle", "quality", require_model=True)  # Hub-style pack: nothing checked locally
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(WEIGHTS_ENV, str(empty))
    with pytest.raises(FileNotFoundError, match="^Missing resources:"):
        model_args("castle", "quality", require_model=True)
    with pytest.raises(FileNotFoundError, match="^Missing resources:"):
        model_args("castle", "realtime", ["--sink-image", str(tmp_path / "nope.png")], require_model=True)


def test_vae_graph_resolution_matches_upstream():
    assert model_args("castle", "realtime", ["--no-vae-graph"]).vae_graph is False
    assert model_args("castle", "realtime", ["--vae-backend", "wan"]).vae_graph is False
    assert model_args("castle", "quality", ["--vae-backend", "taehv", "--vae-graph"]).vae_graph is True
    with pytest.raises(ValueError, match="VAE graph requires TAEHV"):
        model_args("castle", "quality", ["--vae-graph"])
