"""Public scene profiles (MLX port of matrix_game/scenes.py). Weights come only from the MLX pack."""
import argparse
import json
import os
from pathlib import Path

from matrix_game_mlx.inference.loader import WEIGHTS_ENV
from matrix_game_mlx.inference.sink_policy import SINK_ROPE_MODES

SCENES = {
    name: json.loads((Path(__file__).parent / "configs" / (name + ".json")).read_text())
    for name in ("castle", "ancient-egypt", "racing")
}

OPTIMIZED_RUNTIMES = ("rtx5090", "rtx5090-sage", "rtx5080", "rtx5080-sage", "rtx5070ti", "rtx5070ti-sage")
APPLE_RUNTIMES = ("apple", "apple-fast")        # port-only: float16 DiT on the ring-fused runtime


def scene_sink_policy(scene: str) -> tuple:
    """(sink_mode, default sink_rope_mode) exactly as upstream configure_scene."""
    return ("first_frame", "future") if scene == "racing" else ("kv_only", "fixed")


def configure_scene(args):
    from matrix_game_mlx.inference import loader
    profile = SCENES[args.scene]
    args.prompt = getattr(args, "prompt", None) or profile["prompt"]
    args.sink_mode, default_rope_mode = scene_sink_policy(args.scene)
    # --sink-rope-mode overrides the scene default.
    args.sink_rope_mode = getattr(args, "sink_rope_mode", None) or default_rope_mode
    root = args.resources.resolve()
    args.asset_root = args.asset_root or root / ("racing" if args.scene == "racing" else "game")
    if args.sink_image is None:
        args.sink_image = loader.scene_sink_image(args.scene)


def add_model_arguments(parser):
    parser.add_argument("--scene", choices=tuple(SCENES), default="ancient-egypt")
    parser.add_argument("--resources", type=Path, default=Path("resources"))
    for key in ("model-dir", "checkpoint", "taehv-path", "asset-root"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--preset", default="realtime",
                        choices=("realtime", "quality", "baseline") + OPTIMIZED_RUNTIMES + APPLE_RUNTIMES)
    parser.add_argument("--vae-backend", choices=("wan", "taehv"))
    parser.add_argument("--dit-runtime", choices=("native", "kernels", "cudagraph", "ring", "ring-graph", "ring-fused")
                        + OPTIMIZED_RUNTIMES + APPLE_RUNTIMES)
    parser.add_argument("--steps", type=parse_steps)
    parser.add_argument("--first-steps", type=parse_steps)
    parser.add_argument("--vae-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prompt-mode", choices=("fixed", "dynamic"), default="fixed")
    parser.add_argument("--prompt", help="Override the fixed scene caption")
    parser.add_argument("--prompt-interval-chunks", type=int, default=1)
    parser.add_argument("--t5-runtime", choices=("native", "graph"), default="graph")
    parser.add_argument("--sink-image", type=Path)
    parser.add_argument("--sink-rope-mode", choices=SINK_ROPE_MODES,
                        help="Sink RoPE policy; default future for racing, fixed for the shooters")


def parse_steps(value):
    try:
        steps = tuple(int(v) for v in value.split(","))
        if not steps or steps[0] != 1000 or any(not 0 < v <= 1000 for v in steps) or any(a <= b for a, b in zip(steps, steps[1:])):
            raise ValueError
        return steps
    except ValueError:
        raise argparse.ArgumentTypeError("Timesteps must decrease strictly from 1000") from None


def _reject_unsupported(args):
    for flag in ("model_dir", "checkpoint", "taehv_path"):
        if getattr(args, flag) is not None:
            raise ValueError(f"--{flag.replace('_', '-')} is not supported: weights come only from the MLX pack "
                             f"(set {WEIGHTS_ENV} to use a local MLX pack)")
    if args.preset in OPTIMIZED_RUNTIMES:
        raise ValueError(f"--preset {args.preset} is CUDA-only")
    if args.dit_runtime not in (None, "native", "ring-fused") + APPLE_RUNTIMES:
        raise ValueError(f"--dit-runtime {args.dit_runtime} is CUDA-only; the MLX port runs native, ring-fused, apple or apple-fast")


def resolve_model_arguments(args, require_model=True):
    local = os.environ.get(WEIGHTS_ENV)
    _reject_unsupported(args)
    if require_model and local and not (Path(local) / "config.json").is_file():
        raise FileNotFoundError("Missing resources:\n" + str(Path(local) / "config.json"))
    explicit_sink = args.sink_image
    configure_scene(args)
    if args.prompt_mode == "dynamic" and args.scene != "racing":
        raise ValueError("Dynamic captions are currently supported for racing only")
    if not 1 <= args.prompt_interval_chunks <= 1024:
        raise ValueError("Prompt interval must be 1..1024 chunks")
    realtime = args.preset == "realtime" or args.preset in APPLE_RUNTIMES
    if args.preset in APPLE_RUNTIMES:
        args.dit_runtime = args.dit_runtime or args.preset
    args.vae_backend = args.vae_backend or ("taehv" if realtime else "wan")
    args.dit_runtime = args.dit_runtime or ("ring-fused" if args.preset == "realtime" else "native")
    args.steps = args.steps or ((1000,) if realtime else (1000, 500))
    args.first_steps = args.first_steps or args.steps
    if args.vae_graph is None:
        args.vae_graph = realtime and args.vae_backend == "taehv"
    if args.vae_graph and args.vae_backend != "taehv":
        raise ValueError("VAE graph requires TAEHV")
    if require_model:
        files = []
        if explicit_sink is not None:
            files.append(explicit_sink)  # an explicit --sink-image is always checked
        if local:
            pack = Path(local)
            component = json.loads((pack / "config.json").read_text())["scenes"][args.scene]["component"]
            files += [pack / f"{component}.safetensors", pack / "split_model.json"]
            if args.vae_backend == "taehv":
                files.append(pack / "taehv.safetensors")
        files.append(args.sink_image)
        missing = [str(p) for p in dict.fromkeys(files) if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing resources:\n" + "\n".join(missing))
    return args


def model_args(scene: str = "ancient-egypt", preset: str = "quality", argv=(), require_model: bool = False):
    """Parse and resolve model arguments exactly like the CLIs do (single helper for tests and fixtures)."""
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    return resolve_model_arguments(parser.parse_args(["--scene", scene, "--preset", preset, *argv]), require_model)
