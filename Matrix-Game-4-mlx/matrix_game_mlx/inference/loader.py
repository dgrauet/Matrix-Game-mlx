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
