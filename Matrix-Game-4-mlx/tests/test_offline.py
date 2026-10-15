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


@pytest.mark.parametrize("scene,warned", [("castle", True), ("racing", True), ("ancient-egypt", False)])
def test_warns_on_scenes_not_parity_validated(tmp_path, caplog, scene, warned):
    with pytest.raises(FileNotFoundError):
        main(["--scene", scene, "--input", str(tmp_path / "x.mp4"), "--sink-image", str(tmp_path / "missing.png"),
              "--output", str(tmp_path / "out")])
    assert ("not parity-validated" in caplog.text) == warned


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
