"""Model ownership shared by offline inference (MLX port of matrix_game/inference/loader.py)."""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict

from huggingface_hub import hf_hub_download, snapshot_download

PACK_REPO_ID = "dgrauet/matrix-game-4.0-mlx"


def pack_dir() -> Path:
    """Local pack dir from MATRIX_GAME_4_MLX_WEIGHTS_DIR, else the HF snapshot."""
    local = os.environ.get("MATRIX_GAME_4_MLX_WEIGHTS_DIR")
    return Path(local) if local else Path(snapshot_download(PACK_REPO_ID))


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 16 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def resolve_shared_files(pack: Path) -> Dict[str, Path]:
    """Fetch T5/VAE/tokenizer from the pinned MG3 repo (or MATRIX_GAME_4_MLX_SHARED_DIR) and verify sha256."""
    shared = json.loads((Path(pack) / "split_model.json").read_text())["shared"]
    local = os.environ.get("MATRIX_GAME_4_MLX_SHARED_DIR")
    resolved = {}
    for entry in shared["files"]:
        rel = entry["path"]
        path = Path(local) / rel if local else Path(hf_hub_download(shared["repo_id"], rel, revision=shared["revision"]))
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"sha256 mismatch for {rel}: expected {entry['sha256']}, got {digest} ({path})")
        resolved[rel] = path
    out = {name: resolved[name] for name in shared["components"].values()}
    out[shared["tokenizer"]] = resolved[next(r for r in resolved if r.startswith(shared["tokenizer"] + "/"))].parent
    return out


class ModelComponents:
    """Load T5 (encode + free), DiT and VAE for one scene; Wan codec only in milestone 1."""

    def __init__(self, scene: str, preset: str = "quality", sink_image=None, pack: Path = None):
        from matrix_game_mlx.scenes import SCENES, scene_sink_policy
        if preset != "quality":
            raise NotImplementedError("realtime preset (TAEHV): milestone 3")
        self.scene, self.preset = scene, preset
        self.pack = Path(pack) if pack else pack_dir()
        self.cfg = json.loads((self.pack / "config.json").read_text())
        entry = self.cfg["scenes"][scene]
        self.component = entry["component"]
        self.inference = json.loads((self.pack / entry["inference"]).read_text())
        self.sink_image = Path(sink_image) if sink_image else self.pack / entry["sink_image"]
        self.prompt_text = SCENES[scene]["prompt"]
        self.sink_mode, self.sink_rope_mode = scene_sink_policy(scene)
        self.steps = tuple(self.inference["quality_steps"])
        self.gen = self.vae = self.prompt = self.sink_latent = None

    def load(self):
        import gc
        import mlx.core as mx
        from matrix_game_mlx.inference.loading import build_dit
        from matrix_game_mlx.inference.stream import encode_sink_image
        from matrix_game_mlx.inference.wrappers import WanTextEncoder, WanVAE22Wrapper
        shared = resolve_shared_files(self.pack)
        text = WanTextEncoder(str(shared["t5_encoder.safetensors"]), str(shared["google/umt5-xxl"]))
        self.prompt = text([self.prompt_text])["prompt_embeds"].astype(mx.bfloat16)
        mx.eval(self.prompt)
        del text
        gc.collect()
        mx.clear_cache()
        self.gen = build_dit(self.pack, self.component, timestep_shift=self.inference["scheduler_shift"])
        self.vae = WanVAE22Wrapper(str(shared["vae.safetensors"]), dtype=mx.bfloat16)
        self.sink_latent = encode_sink_image(self.vae, self.sink_image, "wan",
                                             first_frame=self.sink_mode == "first_frame").astype(mx.bfloat16)
        mx.eval(self.sink_latent)

    def stream(self, seed: int = 0):
        from matrix_game_mlx.inference.stream import CastleStream
        return CastleStream(self.gen, self.vae, self.prompt, seed=seed, vae_backend="wan", steps=self.steps,
                            first_steps=self.steps, sink_latent=self.sink_latent, sink_mode=self.sink_mode,
                            sink_rope_mode=self.sink_rope_mode)

    def config(self) -> dict:
        return dict(scene=self.scene, component=self.component, preset=self.preset, vae_backend="wan",
                    steps=list(self.steps), sink_mode=self.sink_mode, sink_rope_mode=self.sink_rope_mode,
                    sink_image=self.sink_image.name, prompt=self.prompt_text, dit_runtime="mlx-native")
