import json

import pytest

from matrix_game_mlx.inference.offline import frame_blocks, main


@pytest.mark.parametrize("n,expected", [(1, [(1, 1)]), (2, [(1, 1), (4, 1)]), (5, [(1, 1), (4, 4)]),
                                         (7, [(1, 1), (4, 4), (4, 2)])])
def test_frame_blocks(n, expected):
    blocks = list(frame_blocks(iter(range(n))))
    assert [(len(f), v) for f, v in blocks] == expected
    for frames, valid in blocks:
        assert frames[valid:] == [frames[valid - 1]] * (len(frames) - valid)


def test_refuses_existing_output(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit):
        main(["--input", str(tmp_path / "x.mp4"), "--output", str(tmp_path / "out")])


def test_rejects_incomplete_take(tmp_path):
    take = tmp_path / "take"
    take.mkdir()
    (take / "session.json").write_text(json.dumps({"status": "recording", "frames": 5}))
    with pytest.raises(SystemExit):
        main(["--take", str(take), "--output", str(tmp_path / "out")])


def test_missing_sink_image_fails_before_loading(tmp_path, monkeypatch):
    from matrix_game_mlx.inference import loader
    monkeypatch.setattr(loader, "ModelComponents", lambda *a, **k: pytest.fail("models loaded"))
    sink = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError, match="missing.png"):
        main(["--input", str(tmp_path / "x.mp4"), "--sink-image", str(sink), "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("scene", ["castle", "racing", "ancient-egypt"])
def test_no_parity_warning_for_any_scene(tmp_path, caplog, scene):
    with pytest.raises(FileNotFoundError):
        main(["--scene", scene, "--input", str(tmp_path / "x.mp4"), "--sink-image", str(tmp_path / "missing.png"),
              "--output", str(tmp_path / "out")])
    assert "not parity-validated" not in caplog.text


def _take(tmp_path, **fields):
    from matrix_game_mlx.scenes import SCENES
    take = tmp_path / "take"
    take.mkdir()
    manifest = dict(status="complete", frames=5, width=1280, height=720, fps=24, **fields)
    (take / "session.json").write_text(json.dumps(manifest))
    return take, SCENES


@pytest.mark.parametrize("fields,scene", [
    (dict(scene="castle"), "castle"),
    (dict(game="racing", mode="drive"), "racing"),
    (dict(game="ancient-egypt"), "ancient-egypt"),
])
def test_take_resolves_scene_from_manifest(tmp_path, monkeypatch, fields, scene):
    from matrix_game_mlx.inference import loader
    take, scenes = _take(tmp_path, prompt=None, **fields)
    manifest = json.loads((take / "session.json").read_text())
    manifest["prompt"] = scenes[scene]["prompt"]
    (take / "session.json").write_text(json.dumps(manifest))
    seen = {}

    def fake(scene_name, *a, **k):
        seen["scene"] = scene_name
        raise RuntimeError("stop")
    monkeypatch.setattr(loader, "ModelComponents", fake)
    import imageio_ffmpeg

    class Reader:
        def __next__(self):
            return dict(size=(1280, 720), fps=24)

        def close(self):
            pass
    monkeypatch.setattr(imageio_ffmpeg, "read_frames", lambda *a, **k: Reader())
    with pytest.raises(RuntimeError, match="stop"):
        main(["--take", str(take), "--output", str(tmp_path / "out")])
    assert seen["scene"] == scene


def test_take_prompt_mismatch_rejected(tmp_path):
    take, _ = _take(tmp_path, scene="racing", prompt="wrong")
    with pytest.raises(SystemExit):
        main(["--take", str(take), "--output", str(tmp_path / "out")])


@pytest.mark.needs_weights
def test_render_short_clip(tmp_path):
    import imageio_ffmpeg, numpy as np
    src = tmp_path / "in.mp4"
    w = imageio_ffmpeg.write_frames(str(src), (1280, 720), fps=24, macro_block_size=1)
    w.send(None)
    for i in range(6):
        w.send((np.full((720, 1280, 3), 40 * i, np.uint8)).tobytes())
    w.close()
    main(["--scene", "ancient-egypt", "--input", str(src), "--output", str(tmp_path / "out")])
    out = tmp_path / "out"
    assert (out / "rendered.mp4").is_file() and not (out / "rendered.partial.mp4").exists()
    rep = json.loads((out / "render.json").read_text())
    assert rep["frames"] == 6 and rep["padded_frames"] == 3 and rep["status"] == "complete"
