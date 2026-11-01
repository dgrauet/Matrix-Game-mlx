"""Free-running DiT runtime comparison past the KV window (not collected by pytest).

Run one runtime per process, one process at a time (each loads the real DiT), then compare the dumps:

    python3 -m tests.compare_runtimes --scene castle --preset realtime --chunks 48 --dit-runtime native --out OUT/native
    python3 -m tests.compare_runtimes --scene castle --preset realtime --chunks 48 --dit-runtime ring-fused --out OUT/ring
    python3 -m tests.compare_runtimes --compare OUT/native OUT/ring --json OUT/compare.json

A run streams `--chunks` chunks through `CastleStream.push` (the scene's sink, steps and codec as resolved for the
preset; stream seed `--seed`, so the same noise on every runtime) on a deterministic input: the parity fixtures'
pan (`tests/parity/gen_fixtures.pan_frames`: 480x864 crops of the scene sink image, 8 px per frame, ping-pong),
extended to any length. Each chunk's generated latent and decoded frames are saved (`chunks.npz`: `pred_<i>`
float32, `decoded_<i>` float16 [-1, 1]) with `meta.json`. The stream is free running: each runtime conditions on
its own past outputs, so any per-call difference (here the ring's attention key order) compounds over chunks.

The comparison takes the first directory as the reference and reports per chunk the pred rel L2, the decoded rel L2
and PSNR (peak-to-peak 2, as the parity tests), whether the pred is bitwise equal, and summaries over the chunks
before and from `--window-chunk` (default 21) on.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

PAN_STEP = 8                                     # tests/parity/gen_fixtures.PAN_STEP


def rel_l2(a, b) -> float:
    """||a - b|| / ||b|| in float64 (tests.conftest.rel_l2)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    denom = np.linalg.norm(b)
    return 0.0 if denom == 0 and np.linalg.norm(a) == 0 else float(np.linalg.norm(a - b) / denom)


def psnr(a, b, peak: float = 2.0) -> float:
    """PSNR of [-1, 1] images, peak-to-peak 2 (tests.conftest.psnr)."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(peak ** 2 / mse)


class Pan:
    """gen_fixtures.pan_frames one frame at a time: float32 [3, 480, 864] in [-1, 1] for any frame index."""

    def __init__(self, image_path: Path):
        from PIL import Image
        with Image.open(image_path) as image:
            self.image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = self.image.shape[:2]
        if h < 480 or w < 864:
            raise ValueError(f"{image_path}: {w}x{h} is smaller than one 864x480 frame")
        self.top, self.span = (h - 480) // 2, w - 864

    def frame(self, i: int) -> np.ndarray:
        x = (i * PAN_STEP) % (2 * self.span) if self.span else 0
        left = x if x <= self.span else 2 * self.span - x
        crop = self.image[self.top:self.top + 480, left:left + 864]
        return (crop.transpose(2, 0, 1).astype(np.float32) * (2.0 / 255.0) - 1.0).astype(np.float32)


def run(args) -> dict:
    import mlx.core as mx
    from matrix_game_mlx.inference.loader import ModelComponents
    from matrix_game_mlx.scenes import resolve_model_arguments
    resolve_model_arguments(args)
    if args.sink_image is None:
        raise SystemExit("The input pan needs the scene sink image (do not pass --no-sink-image)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    comp = ModelComponents(args)
    comp.load()
    if comp.dit_runtime() != args.dit_runtime:
        raise SystemExit(f"Installed DiT runtime {comp.dit_runtime()} != requested {args.dit_runtime}")
    pan = Pan(args.sink_image)
    t0 = time.perf_counter()
    stream = comp.stream(seed=args.seed)
    window = comp.gen.model.blocks[0].self_attn.local_attn_size
    arrays, chunks, frame = {}, [], 0
    for i in range(args.chunks):
        count = 1 if i == 0 else 4
        pixels = mx.array(np.stack([pan.frame(frame + k) for k in range(count)]))
        position = stream.index + stream.sink_offset
        chunk = stream.push(pixels, frame_start=frame)
        arrays[f"pred_{i}"] = np.array(chunk.latent.astype(mx.float32))
        arrays[f"decoded_{i}"] = np.array(chunk.pixels.astype(mx.float32)).astype(np.float16)
        chunks.append({"index": i, "position": position, "frames": [frame, frame + count - 1],
                       "denoising_steps": chunk.denoising_steps})
        frame += count
        print(f"chunk {i} position {position} {time.perf_counter() - t0:.1f} s", flush=True)
    runtime = getattr(comp.gen.model, "castle_runtime", None)
    counters = None if runtime is None else {"captures": runtime.captures, "replays": runtime.replays}
    stream.close()
    np.savez(out / "chunks.npz", **arrays)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        commit = None
    meta = {"scene": args.scene, "preset": args.preset, "dit_runtime": comp.dit_runtime(), "chunks": args.chunks,
            "seed": args.seed, "sink_mode": args.sink_mode, "sink_rope_mode": args.sink_rope_mode,
            "sink_offset": stream.sink_offset, "window": window, "steps": list(args.steps),
            "first_steps": list(args.first_steps), "vae_backend": args.vae_backend,
            "vae_graph": getattr(args, "vae_graph", None), "input": f"pan of {args.sink_image.name}, {PAN_STEP} px/frame",
            "runtime_counters": counters, "seconds": round(time.perf_counter() - t0, 1),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2), "mlx": mx.__version__, "commit": commit,
            "per_chunk": chunks}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "per_chunk"}))
    return meta


def _summary(rows) -> dict:
    if not rows:
        return {}
    preds, psnrs = [r["pred_rel_l2"] for r in rows], [r["decoded_psnr"] for r in rows]
    finite = [p for p in psnrs if p != float("inf")]
    return {"chunks": [rows[0]["index"], rows[-1]["index"]], "count": len(rows),
            "pred_bitwise": sum(r["pred_bitwise"] for r in rows),
            "pred_rel_l2_median": statistics.median(preds), "pred_rel_l2_max": max(preds),
            "decoded_psnr_median": statistics.median(psnrs), "decoded_psnr_min": min(psnrs),
            "decoded_psnr_finite": len(finite)}


def compare(reference: Path, other: Path, window_chunk: int, json_path=None) -> dict:
    meta_a, meta_b = (json.loads((Path(d) / "meta.json").read_text()) for d in (reference, other))
    for key in ("scene", "preset", "chunks", "seed", "sink_mode", "steps", "input"):
        if meta_a[key] != meta_b[key]:
            raise SystemExit(f"Runs differ in {key}: {meta_a[key]} vs {meta_b[key]}")
    a, b = np.load(Path(reference) / "chunks.npz"), np.load(Path(other) / "chunks.npz")
    rows = []
    for i, chunk in enumerate(meta_a["per_chunk"]):
        pa, pb = a[f"pred_{i}"], b[f"pred_{i}"]
        da, db = a[f"decoded_{i}"].astype(np.float32), b[f"decoded_{i}"].astype(np.float32)
        rows.append({"index": i, "position": chunk["position"], "pred_bitwise": bool(np.array_equal(pa, pb)),
                     "pred_rel_l2": rel_l2(pb, pa), "decoded_rel_l2": rel_l2(db, da), "decoded_psnr": psnr(db, da)})
    result = {"reference": meta_a["dit_runtime"], "other": meta_b["dit_runtime"], "scene": meta_a["scene"],
              "preset": meta_a["preset"], "chunks": meta_a["chunks"], "window": meta_a["window"],
              "sink_offset": meta_a["sink_offset"], "window_chunk": window_chunk,
              "first_non_bitwise_chunk": next((r["index"] for r in rows if not r["pred_bitwise"]), None),
              "before_window": _summary([r for r in rows if r["index"] < window_chunk]),
              "from_window": _summary([r for r in rows if r["index"] >= window_chunk]),
              "per_chunk": rows, "runs": {"reference": {k: v for k, v in meta_a.items() if k != "per_chunk"},
                                          "other": {k: v for k, v in meta_b.items() if k != "per_chunk"}}}
    print(f"{result['scene']} {result['preset']}: {result['other']} vs {result['reference']} "
          f"(window {result['window']}, sink offset {result['sink_offset']})")
    print("| chunk | position | pred bitwise | pred rel L2 | decoded rel L2 | decoded PSNR (dB) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['index']} | {r['position']} | {'yes' if r['pred_bitwise'] else 'no'} | {r['pred_rel_l2']:.3g} | "
              f"{r['decoded_rel_l2']:.3g} | {r['decoded_psnr']:.2f} |")
    for name in ("before_window", "from_window"):
        print(name, json.dumps(result[name]))
    if json_path:
        Path(json_path).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    from matrix_game_mlx.scenes import add_model_arguments
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_model_arguments(parser)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="Run mode: output directory (chunks.npz, meta.json)")
    parser.add_argument("--compare", nargs=2, metavar=("REFERENCE", "OTHER"), help="Compare two run directories")
    parser.add_argument("--window-chunk", type=int, default=21, help="First chunk of the 'from window' summary")
    parser.add_argument("--json", help="Compare mode: write the result here")
    parser.set_defaults(scene="castle", preset="realtime", resources=Path(os.environ.get("MG4_RESOURCES", "resources")))
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare, args.window_chunk, args.json)
    if not args.out:
        parser.error("--out is required to run (or pass --compare REFERENCE OTHER)")
    return run(args)


if __name__ == "__main__":
    main()
