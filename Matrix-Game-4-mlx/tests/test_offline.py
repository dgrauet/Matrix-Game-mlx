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


def test_missing_sink_image_fails_before_loading(tmp_path, monkeypatch, capsys):
    from matrix_game_mlx.inference import loader
    monkeypatch.setattr(loader, "ModelComponents", lambda *a, **k: pytest.fail("models loaded"))
    sink = tmp_path / "missing.png"
    with pytest.raises(SystemExit):
        main(["--input", str(tmp_path / "x.mp4"), "--sink-image", str(sink), "--output", str(tmp_path / "out")])
    assert "missing.png" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("scene", ["castle", "racing", "ancient-egypt"])
def test_no_parity_warning_for_any_scene(tmp_path, caplog, scene):
    with pytest.raises(SystemExit):
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

    def fake(args, *a, **k):
        seen["scene"] = args.scene
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


def test_preset_option(monkeypatch, tmp_path):
    from matrix_game_mlx.inference import loader
    seen = []

    def fake(args, *a, **k):
        seen.append(args.preset)
        raise RuntimeError("stop")
    monkeypatch.setattr(loader, "ModelComponents", fake)
    import imageio_ffmpeg

    class Reader:
        def __next__(self):
            return dict(size=(1280, 720), fps=24)

        def close(self):
            pass
    monkeypatch.setattr(imageio_ffmpeg, "read_frames", lambda *a, **k: Reader())
    for extra, expected in (([], "quality"), (["--preset", "realtime"], "realtime")):
        with pytest.raises(RuntimeError, match="stop"):
            main(["--input", str(tmp_path / "x.mp4"), "--output", str(tmp_path / "out"), *extra])
        assert seen[-1] == expected
    with pytest.raises(SystemExit):
        main(["--input", str(tmp_path / "x.mp4"), "--output", str(tmp_path / "out"), "--preset", "bogus"])


# ---- full upstream render surface (model-free fakes) ----

class _FakeStream:
    def __init__(self):
        self.prompts, self.pushes, self.closed = [], [], False

    def set_prompt(self, embeddings):
        self.prompts.append((len(self.pushes), embeddings))

    def push(self, pixels, frame_start=0):
        import types
        self.pushes.append(frame_start)
        return types.SimpleNamespace(pixels=pixels)

    def close(self):
        self.closed = True


class _FakeComponents:
    last = None

    def __init__(self, args, pack=None):
        self.args = args
        self.caption_encoder = (lambda text: "emb:" + text) if args.prompt_mode == "dynamic" else None
        self.fake_stream = _FakeStream()
        self.stream_seed = None
        _FakeComponents.last = self

    def load(self):
        pass

    def stream(self, **kwargs):
        self.stream_seed = kwargs.get("seed")
        return self.fake_stream

    def config(self):
        return dict(scene=self.args.scene, prompt=self.args.prompt, prompt_mode=self.args.prompt_mode)


@pytest.fixture
def fake_components(monkeypatch):
    from matrix_game_mlx.inference import loader
    monkeypatch.setattr(loader, "ModelComponents", _FakeComponents)
    return _FakeComponents


def _video(tmp_path, frames=9, size=(128, 72)):
    import imageio_ffmpeg
    import numpy as np
    src = tmp_path / "in.mp4"
    w = imageio_ffmpeg.write_frames(str(src), size, fps=24, macro_block_size=1)
    w.send(None)
    for i in range(frames):
        w.send(np.full((size[1], size[0], 3), 20 * i, np.uint8).tobytes())
    w.close()
    return src


def _captions(tmp_path, prompts, frames=9):
    """One record per chunk: 9 frames -> chunks of 1, 4, 4."""
    ranges, start = [], 0
    for count in [1, 4, 4, 4, 4][:len(prompts)]:
        ranges.append((start, start + count - 1))
        start += count
    path = tmp_path / "captions.jsonl"
    path.write_text("".join(json.dumps(dict(frame_start=a, frame_end=b, prompt=p)) + "\n"
                            for (a, b), p in zip(ranges, prompts)))
    return path


def _dynamic(tmp_path, captions, *extra):
    return ["--scene", "racing", "--prompt-mode", "dynamic", "--input", str(_video(tmp_path)),
            "--captions", str(captions), "--output", str(tmp_path / "out"), *extra]


def test_dynamic_requires_captions(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--scene", "racing", "--prompt-mode", "dynamic", "--input", str(tmp_path / "x.mp4"),
              "--output", str(tmp_path / "out")])
    assert "Offline dynamic mode requires --captions; an RGB video alone contains no engine captions" in capsys.readouterr().err


def test_captions_require_dynamic(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--input", str(tmp_path / "x.mp4"), "--captions", str(tmp_path / "c.jsonl"),
              "--output", str(tmp_path / "out")])
    assert "--captions requires --prompt-mode dynamic" in capsys.readouterr().err


def test_resolve_errors_become_parser_errors(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--dit-runtime", "rtx5090", "--input", str(tmp_path / "x.mp4"), "--output", str(tmp_path / "out")])
    assert "CUDA-only" in capsys.readouterr().err


def test_caption_ranges_must_match_chunks(tmp_path, fake_components):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(dict(frame_start=1, frame_end=1, prompt="A")) + "\n")
    with pytest.raises(ValueError, match="Caption ranges must match consecutive decoded RGB chunks"):
        main(_dynamic(tmp_path, bad))
    assert fake_components.last.fake_stream.closed


def test_invalid_caption_rejected(tmp_path, fake_components):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(dict(frame_start=0, frame_end=0, prompt="  ")) + "\n")
    with pytest.raises(ValueError, match="Invalid caption"):
        main(_dynamic(tmp_path, bad))


def test_prompt_switch_at_interval(tmp_path, fake_components):
    main(_dynamic(tmp_path, _captions(tmp_path, ["A", "A", "B"]), "--prompt-interval-chunks", "1"))
    assert fake_components.last.fake_stream.prompts == [(0, "emb:A"), (2, "emb:B")]
    assert fake_components.last.stream_seed == 0


def test_prompt_switch_respects_interval_two(tmp_path, fake_components):
    main(_dynamic(tmp_path, _captions(tmp_path, ["A", "B", "C"]), "--prompt-interval-chunks", "2"))
    assert fake_components.last.fake_stream.prompts == [(0, "emb:A"), (2, "emb:C")]


def test_captions_beyond_video(tmp_path, fake_components):
    with pytest.raises(ValueError, match="Caption file extends beyond the input video"):
        main(_dynamic(tmp_path, _captions(tmp_path, ["A", "A", "A", "A"])))


def test_captions_shorter_than_video(tmp_path, fake_components):
    with pytest.raises(ValueError):
        main(_dynamic(tmp_path, _captions(tmp_path, ["A", "A"])))
    assert not (tmp_path / "out" / "rendered.mp4").exists()


def test_take_prompt_mismatch_rejected_even_with_prompt_override(tmp_path, capsys):
    take, _ = _take(tmp_path, scene="racing", prompt="wrong")
    with pytest.raises(SystemExit):
        main(["--take", str(take), "--prompt", "x", "--output", str(tmp_path / "out")])
    assert "Recording scene/prompt does not match release profiles" in capsys.readouterr().err


def test_report_fields(tmp_path, fake_components):
    main(["--input", str(_video(tmp_path)), "--output", str(tmp_path / "out"), "--seed", "7"])
    out = tmp_path / "out"
    assert (out / "rendered.mp4").is_file() and not (out / "rendered.partial.mp4").exists()
    rep = json.loads((out / "render.json").read_text())
    for key in ("scene", "frames", "padded_frames", "fps", "seed", "status", "source_name", "seconds",
                "seconds_per_chunk", "peak_memory_gb", "mlx"):
        assert key in rep
    assert "torch" not in rep
    assert rep["frames"] == 9 and rep["seed"] == 7 and rep["status"] == "complete"
    assert fake_components.last.stream_seed == 7
