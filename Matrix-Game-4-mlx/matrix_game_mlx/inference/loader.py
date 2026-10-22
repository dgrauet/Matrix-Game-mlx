"""Model ownership shared by offline inference (MLX port of matrix_game/inference/loader.py)."""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict

from huggingface_hub import hf_hub_download, snapshot_download

PACK_REPO_ID = "dgrauet/matrix-game-4.0-mlx"
WEIGHTS_ENV = "MATRIX_GAME_4_MLX_WEIGHTS_DIR"
PACK_REVISION = "327ce1f7ae729224ba37b0326fbf0eb4f79e0503"


def pack_dir(components=None) -> Path:
    """Pack dir: MATRIX_GAME_4_MLX_WEIGHTS_DIR if set, else the pinned Hub pack (PACK_REPO_ID at PACK_REVISION).

    From the Hub, the small files (json configs, sink images) are always fetched, plus the `components` safetensors
    (e.g. ("dit_castle", "taehv")); default is every component. Downloads are cached by huggingface_hub."""
    local = os.environ.get(WEIGHTS_ENV)
    if local:
        return Path(local)
    weights = ["*.safetensors"] if components is None else [f"{c}.safetensors" for c in components]
    return Path(snapshot_download(PACK_REPO_ID, revision=PACK_REVISION, allow_patterns=["*.json", "*.png", *weights]))


def scene_sink_image(scene: str) -> Path:
    """Pack sink image path for `scene`, read from the pack's config.json (no weights are downloaded)."""
    pack = pack_dir(())
    return pack / json.loads((pack / "config.json").read_text())["scenes"][scene]["sink_image"]


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 16 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def resolve_shared_files(pack: Path, components=None) -> Dict[str, Path]:
    """Fetch T5/VAE/tokenizer from the pinned MG3 repo (or MATRIX_GAME_4_MLX_SHARED_DIR) and verify sha256.

    `components` limits the resolved component files to those names (keys of the pack's shared components, e.g.
    {"t5_encoder"} for the TAEHV realtime preset, which does not need the Wan VAE); default is all. The tokenizer is
    always resolved, and every resolved file is sha-checked."""
    shared = json.loads((Path(pack) / "split_model.json").read_text())["shared"]
    local = os.environ.get("MATRIX_GAME_4_MLX_SHARED_DIR")
    wanted = shared["components"] if components is None else {k: shared["components"][k] for k in components}
    skipped = set(shared["components"].values()) - set(wanted.values())
    resolved = {}
    for entry in shared["files"]:
        rel = entry["path"]
        if rel in skipped:
            continue
        path = Path(local) / rel if local else Path(hf_hub_download(shared["repo_id"], rel, revision=shared["revision"]))
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"sha256 mismatch for {rel}: expected {entry['sha256']}, got {digest} ({path})")
        resolved[rel] = path
    out = {name: resolved[name] for name in wanted.values()}
    out[shared["tokenizer"]] = resolved[next(r for r in resolved if r.startswith(shared["tokenizer"] + "/"))].parent
    return out


class ModelComponents:
    """Model ownership for one resolved CLI/API `args` (see matrix_game_mlx.scenes.resolve_model_arguments).

    Loads the DiT, the codec (Wan VAE or TAEHV, chosen by `args.vae_backend`) and the prompt embeddings. Fixed prompts
    are encoded once and T5 is freed; dynamic prompts keep a resident CaptionEncoder."""

    def __init__(self, args, pack: Path = None):
        self.args = args
        self.pack = Path(pack) if pack else pack_dir(())
        self.cfg = json.loads((self.pack / "config.json").read_text())
        self.entry = self.cfg["scenes"][args.scene]
        self.component = self.entry["component"]
        self.gen = self.vae = self.prompt = None
        self.sink_latent = self.sink_sha256 = None
        self.attention_backend = "mlx-sdpa"
        self.caption_encoder = None
        self._fetch_weights = pack is None

    def load(self):
        import gc
        import mlx.core as mx
        from matrix_game_mlx.inference.loading import build_dit
        from matrix_game_mlx.inference.stream import encode_sink_image
        from matrix_game_mlx.inference.text import CaptionEncoder
        from matrix_game_mlx.inference.wrappers import WanTextEncoder
        args = self.args
        taehv = args.vae_backend == "taehv"
        if self._fetch_weights:
            self.pack = pack_dir((self.component, "taehv") if taehv else (self.component,))
        shared = resolve_shared_files(self.pack, ("t5_encoder",) if taehv else ("t5_encoder", "vae"))
        inference = json.loads((self.pack / self.entry["inference"]).read_text())
        text = WanTextEncoder(str(shared["t5_encoder.safetensors"]), str(shared["google/umt5-xxl"]))
        if args.prompt_mode == "dynamic":
            self.caption_encoder = CaptionEncoder(text, graph=args.t5_runtime == "graph")
            self.prompt = self.caption_encoder(args.prompt)
        else:
            self.prompt = text([args.prompt])["prompt_embeds"].astype(mx.bfloat16)
            mx.eval(self.prompt)
        del text
        gc.collect()
        mx.clear_cache()
        self.gen = build_dit(self.pack, self.component, timestep_shift=inference["scheduler_shift"])
        if taehv:
            from matrix_game_mlx.inference.taehv import TAEHV
            self.vae = TAEHV.from_pack(self.pack / "taehv.safetensors", dtype=mx.bfloat16)
        else:
            from matrix_game_mlx.inference.wrappers import WanVAE22Wrapper
            self.vae = WanVAE22Wrapper(str(shared["vae.safetensors"]), dtype=mx.bfloat16)
        if args.sink_image is not None:
            self.sink_latent = encode_sink_image(self.vae, args.sink_image, args.vae_backend,
                                                 first_frame=args.sink_mode == "first_frame").astype(mx.bfloat16)
            mx.eval(self.sink_latent)
            self.sink_sha256 = sha256_file(args.sink_image)

    def stream(self, **kwargs):
        from matrix_game_mlx.inference.stream import CastleStream
        return CastleStream(self.gen, self.vae, self.prompt, vae_backend=self.args.vae_backend,
                            steps=self.args.steps, first_steps=self.args.first_steps,
                            sink_latent=self.sink_latent, sink_mode=self.args.sink_mode,
                            sink_rope_mode=self.args.sink_rope_mode, **kwargs)

    def config(self):
        from matrix_game_mlx.scenes import SCENES
        a = self.args
        return dict(scene=a.scene, scene_name=SCENES[a.scene]["name"], checkpoint=f"{self.component}.safetensors",
                    ckpt_keys=["generator_ema"], vae_backend=a.vae_backend, steps=list(a.steps),
                    first_steps=list(a.first_steps), sink_rope_mode=a.sink_rope_mode, sink_mode=a.sink_mode,
                    sink_encoder=a.vae_backend, sink_source="image" if a.sink_image is not None else "generated_first_frame",
                    sink_image=a.sink_image.name if a.sink_image is not None else None, sink_sha256=self.sink_sha256,
                    prompt=a.prompt, hq_first_frame=a.sink_mode == "first_frame" and a.sink_image is not None,
                    prompt_mode=a.prompt_mode, prompt_interval_chunks=a.prompt_interval_chunks,
                    t5_runtime=a.t5_runtime, dit_parallel="single", dit_runtime=a.dit_runtime,
                    attention_backend=self.attention_backend, vae_graph=False, early_publish=False,
                    profile_chunks=False, reference_pipeline="stream", experimental_runtime=False,
                    preset=a.preset, component=self.component)
