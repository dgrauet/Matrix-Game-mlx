"""Public scene profiles (MLX port of matrix_game/scenes.py, model-loading parts only)."""
import json
from pathlib import Path

SCENES = {
    name: json.loads((Path(__file__).parent / "configs" / (name + ".json")).read_text())
    for name in ("castle", "ancient-egypt", "racing")
}


def scene_sink_policy(scene: str) -> tuple:
    """(sink_mode, sink_rope_mode) exactly as upstream configure_scene."""
    return ("first_frame", "absolute") if scene == "racing" else ("kv_only", "fixed")
