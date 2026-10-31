"""Record a real online (serve-mode) game capture for the *_online_* parity variants (gen_fixtures.VARIANT_SPECS).

Run manually from Matrix-Game-4-mlx/ (not collected by pytest), with MG4_RESOURCES set:
    python -m tests.parity.capture_online --scene racing|castle [--out DIR]

The capture runs the real native game server exactly as `matrix-game-mlx serve` does for a session (RacingGame /
CastleGame with ReferencePreprocessor(model=True), --fps 24), sends the committed scripted controls
(tests/data/online/<scene>_controls.json: message i, validated by the serve controls schema, is sent before waiting
for chunk i) and keeps gen_fixtures.VARIANT_CHUNKS chunks. It writes, in DIR (default
$MG4_RESOURCES/online-captures/<scene>-online/, git-ignored):
- frames.npy: uint8 [N, 480, 864, 3], N = 1 + 4 (chunks - 1), the RGB images the online producer feeds
  frames_to_tensor (PreparedReference.images);
- states.jsonl: one line per chunk (frame_start, states, and the engine caption chunk_prompt for racing);
- capture.json: scene, fps, seed, game options, controls file and sha256, binary sha256, frames_sha256, hello.

Racing runs in free drive (time_trial holds the car for its countdown, so the caption would never change): the
throttle makes the engine caption's speed clause change between gated chunks. The racing capture must show an
engine-caption change between two consecutive gated chunks after chunk 0 (not only the chunk-0 replacement of the
profile prompt); the command exits non-zero otherwise.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from tests.parity.gen_fixtures import (VARIANT_CHUNKS, VARIANT_SPECS, capture_dir, capture_frames, file_sha256,
                                       gated_chunks)
from tests.racing_game import UPSTREAM, castle_game_binary, racing_server_binary, resources_dir

FPS = 24
CONTROLS_DIR = Path(__file__).resolve().parents[1] / "data" / "online"
CASTLE_SEED = 1729                                   # CastleGame default: serve passes no seed
RACING_OPTIONS = dict(mode="drive", route="downtown")
VARIANTS = {"racing": "racing_online_dynamic", "castle": "castle_online_realtime"}


def gated_caption_change(captions: list, gated: tuple) -> Optional[int]:
    """The first gated chunk k >= 1 whose engine caption differs from gated chunk k - 1's, else None."""
    return next((k for k in gated if k >= 1 and k - 1 in gated and captions[k] != captions[k - 1]), None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--out", type=Path, help="Capture directory (default $MG4_RESOURCES/online-captures/<scene>-online)")
    args = parser.parse_args(argv)
    args.out = args.out or capture_dir(VARIANT_SPECS[VARIANTS[args.scene]].capture)
    return args


def open_game(scene: str, log_path: Path):
    """The game as serve's Session.start opens it (stream reference pipeline, model mode)."""
    from matrix_game_mlx.scenes import SCENES
    from matrix_game_mlx.server.reference_input import ReferencePreprocessor
    prepare = ReferencePreprocessor(model=True)
    if scene == "racing":
        from matrix_game_mlx.server.racing import RacingGame
        return RacingGame(UPSTREAM / "game", fps=FPS, asset_root=resources_dir() / "racing", options=RACING_OPTIONS,
                          prepare=prepare, log_path=log_path)
    from matrix_game_mlx.server.game import CastleGame
    return CastleGame(UPSTREAM / "game", fps=FPS, asset_root=resources_dir() / "game", scene=scene,
                      expected_prompt=SCENES[scene]["prompt"], seed=CASTLE_SEED, prepare=prepare, log_path=log_path)


def capture(scene: str, out: Path, chunks: int = VARIANT_CHUNKS) -> dict:
    """Record the capture into `out` (module docstring); returns capture.json's content."""
    if scene == "racing":
        from matrix_game_mlx.server.racing import controls as validate
    else:
        from matrix_game_mlx.server.app import controls as validate
    controls_file = CONTROLS_DIR / f"{scene}_controls.json"
    messages = [validate(m) for m in json.loads(controls_file.read_text())]
    if len(messages) < chunks:
        raise ValueError(f"{controls_file}: {len(messages)} control messages, need one per chunk ({chunks})")
    out.mkdir(parents=True, exist_ok=True)
    game = open_game(scene, out / "game.log")
    images, pixels, rows = [], [], []
    try:
        cancel = threading.Event()
        for i in range(chunks):
            game.send(messages[i])
            value = game.next(cancel)
            if value is None:
                raise RuntimeError(f"Game ended before chunk {i}")
            reference, states = value
            images += [np.asarray(image, dtype=np.uint8) for image in reference.images]
            pixels.append(np.array(reference.pixels))
            row = dict(frame_start=states[0]["frame"], states=states)
            if scene == "racing":
                row["chunk_prompt"] = states[-1]["chunk_prompt"]
            rows.append(row)
        hello = game.hello
    finally:
        game.close()
    np.save(out / "frames.npy", np.stack(images))
    (out / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    binary = racing_server_binary() if scene == "racing" else castle_game_binary()
    v = VARIANT_SPECS[VARIANTS[scene]]
    meta = dict(scene=scene, variant=v.name, fps=FPS, chunks=chunks, frames=len(images),
                seed=None if scene == "racing" else CASTLE_SEED,
                options=hello.get("game_options") if scene == "racing" else None,
                controls_file=str(controls_file.relative_to(CONTROLS_DIR.parents[2])),
                controls_sha256=file_sha256(controls_file), binary=str(binary.relative_to(UPSTREAM.parent)),
                binary_sha256=file_sha256(binary), frames_sha256=file_sha256(out / "frames.npy"),
                reference_pipeline="stream (ReferencePreprocessor(model=True), as serve)", hello=hello,
                recorded=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if scene == "racing":
        gated = gated_chunks(v)
        meta.update(gated_chunks=list(gated),
                    caption_change_chunk=gated_caption_change([r["chunk_prompt"] for r in rows], gated))
    (out / "capture.json").write_text(json.dumps(meta, indent=2))
    # The saved frames must give back exactly the pixels the online producer pushed
    if not np.array_equal(capture_frames(out, len(images)), np.concatenate(pixels)):
        raise RuntimeError("frames.npy does not reproduce the producer's frames_to_tensor pixels")
    return meta


def main(argv=None):
    args = parse_args(argv)
    meta = capture(args.scene, args.out)
    print(f"{args.out}: {meta['frames']} frames, frames_sha256 {meta['frames_sha256']}")
    if args.scene == "racing":
        rows = [json.loads(line) for line in (args.out / "states.jsonl").read_text().splitlines()]
        for i, row in enumerate(rows):
            print(f"chunk {i} caption: {row['chunk_prompt']!r}")
        if meta["caption_change_chunk"] is None:
            sys.exit(f"No engine-caption change between two gated chunks {meta['gated_chunks']} after chunk 0: "
                     "change the controls / route so the caption changes")
        print(f"engine caption changes at gated chunk {meta['caption_change_chunk']}")


if __name__ == "__main__":
    main()
