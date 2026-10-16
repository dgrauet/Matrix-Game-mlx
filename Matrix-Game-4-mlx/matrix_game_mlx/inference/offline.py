"""Render a video or a complete local recording using the same stream as online (MLX)."""
import argparse
import json
from pathlib import Path
import time

from matrix_game_mlx.scenes import SCENES


def frame_blocks(reader):
    """Yield (frames, valid): one frame, then blocks of 4; the last block is padded with its last frame."""
    count = 1
    while True:
        frames = []
        for _ in range(count):
            value = next(reader, None)
            if value is None:
                break
            frames.append(value)
        if not frames:
            return
        valid = len(frames)
        frames.extend([frames[-1]] * (count - valid))
        yield frames, valid
        if valid < count:
            return
        count = 4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(SCENES), default="ancient-egypt")
    parser.add_argument("--preset", choices=("quality", "realtime"), default="quality")
    parser.add_argument("--sink-image", type=Path, help="Override the scene sink image")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Clean low-poly RGB video")
    source.add_argument("--take", type=Path, help="Completed game recording directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists")
    manifest = None
    if args.take:
        manifest = json.loads((args.take / "session.json").read_text())
        if manifest.get("status") != "complete" or manifest.get("frames", 0) < 1:
            parser.error("Expected a complete nonempty recording")
        scene = manifest.get("scene", manifest.get("game"))
        if args.scene == "racing" and manifest.get("mode") in ("drive", "time_trial", "pursuit"):
            scene = "racing"
        if scene not in SCENES or manifest.get("prompt") != SCENES[scene]["prompt"]:
            parser.error("Recording scene/prompt does not match release profiles")
        args.scene, args.input = scene, args.take / "reference.mkv"
    if args.sink_image is not None and not args.sink_image.is_file():
        raise FileNotFoundError(f"Missing resources:\n{args.sink_image}")
    import imageio_ffmpeg
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from .loader import ModelComponents
    from .preprocessing import crop_and_resize, frames_to_tensor, tensor_to_uint8

    reader = imageio_ffmpeg.read_frames(str(args.input), pix_fmt="rgb24")
    writer = stream = None
    try:
        info = next(reader)
        width, height = info["size"]
        fps = info["fps"]
        if manifest and ((width, height) != (manifest["width"], manifest["height"]) or abs(fps - manifest["fps"]) > .001):
            raise ValueError("Recording dimensions or FPS differ from manifest")
        components = ModelComponents(args.scene, args.preset, args.sink_image)
        components.load()
        stream = components.stream(seed=args.seed)
        args.output.mkdir(parents=True)
        partial = args.output / "rendered.partial.mp4"
        writer = imageio_ffmpeg.write_frames(str(partial), (864, 480), fps=fps, codec="libx264",
                    pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
                    output_params=["-crf", "16", "-preset", "fast", "-movflags", "+faststart"])
        writer.send(None)
        written, padded, chunks = 0, 0, 0
        started = time.perf_counter()
        for index, (raw, valid) in enumerate(frame_blocks(reader)):
            frames = [crop_and_resize(Image.fromarray(np.frombuffer(r, np.uint8).reshape(height, width, 3)), 480, 864) for r in raw]
            result = stream.push(frames_to_tensor(frames), frame_start=written)
            rgb = tensor_to_uint8(result.pixels)[:valid]
            if len(rgb) != valid:
                raise ValueError("Decoder output cadence mismatch")
            for frame in rgb:
                writer.send(frame.tobytes())
            written += valid
            padded += len(raw) - valid
            chunks += 1
            if index % 25 == 0:
                print(f"{written} frames | {written / (time.perf_counter() - started):.2f} processing FPS", flush=True)
        if written == 0 or (manifest and written != manifest["frames"]):
            raise ValueError("Input frame count differs from recording manifest")
        writer.close()
        writer = None
        # Never label a truncated encoder output as a completed rendering.
        check = imageio_ffmpeg.read_frames(str(partial), pix_fmt="rgb24")
        try:
            encoded = next(check)
            count = sum(1 for _ in check)
        finally:
            check.close()
        if count != written or encoded["size"] != (864, 480) or abs(encoded["fps"] - fps) > .01:
            raise RuntimeError("Encoded output failed frame-count verification")
        partial.rename(args.output / "rendered.mp4")
        seconds = time.perf_counter() - started
        report = dict(components.config(), frames=written, padded_frames=padded, fps=fps,
                      seed=args.seed, status="complete", source_name=args.input.name,
                      seconds=seconds, seconds_per_chunk=seconds / chunks,
                      peak_memory_gb=mx.get_peak_memory() / 1e9, mlx=mx.__version__)
        (args.output / "render.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Render complete:", args.output / "rendered.mp4")
    finally:
        reader.close()
        if writer is not None:
            writer.close()
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
