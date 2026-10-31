"""DiT speed benchmark (not collected by pytest): python3 -m tests.bench_dit --scene castle --preset realtime --chunks 14."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import mlx.core as mx

from matrix_game_mlx.inference.loader import ModelComponents
from matrix_game_mlx.scenes import add_model_arguments, resolve_model_arguments

STAGES = ("encode", "denoise", "clean_kv", "decode")


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--chunks", type=int, default=14)
    parser.add_argument("--json", default=None)
    parser.set_defaults(scene="castle", preset="realtime", resources=Path(os.environ.get("MG4_RESOURCES", "resources")))
    args = parser.parse_args(argv)
    resolve_model_arguments(args)
    comp = ModelComponents(args)
    comp.load()
    t0 = time.perf_counter()
    stream = comp.stream(seed=0, profile=True)        # includes the runtime's warm-up (prime) and the sink pass
    init_ms = (time.perf_counter() - t0) * 1000
    window = comp.gen.model.blocks[0].self_attn.local_attn_size
    key, frame, chunks, steady = mx.random.key(1), 0, [], []
    for i in range(args.chunks + 1):
        count = 1 if i == 0 else 4
        key, sub = mx.random.split(key)
        pixels = mx.random.uniform(-1, 1, (count, 3, 480, 864), key=sub)
        steady.append(stream.index + stream.sink_offset > window)   # full KV window (compiled graph on ring-fused)
        timings = stream.push(pixels, frame_start=frame).timings_ms
        frame += count
        chunks.append({name: timings[name] for name in STAGES})
        print(i, {k: round(v) for k, v in chunks[-1].items()}, flush=True)
    rest = chunks[1:]
    totals = [sum(c.values()) for c in rest]
    steady_totals = [sum(c.values()) for c, s in zip(rest, steady[1:]) if s]
    result = {
        "scene": args.scene, "preset": args.preset, "dit_runtime": args.dit_runtime, "chunks": args.chunks,
        "stream_init_ms": round(init_ms),
        "first_chunk_ms": round(sum(chunks[0].values())),
        "first_frame_ms": round(init_ms + sum(chunks[0].values())),
        "stage_ms_median": {n: round(statistics.median(c[n] for c in rest)) for n in STAGES},
        "chunk_ms_median": round(statistics.median(totals)),
        "chunk_ms_last4": round(statistics.mean(totals[-4:])),
        "steady_chunks": len(steady_totals),
        "chunk_ms_steady_median": round(statistics.median(steady_totals)) if steady_totals else None,
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
        "mlx": mx.__version__,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)
    return result


if __name__ == "__main__":
    main()
