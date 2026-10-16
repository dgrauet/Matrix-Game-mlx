"""Cheap checks of the fixture generator's helpers (no weights, no model build)."""
import importlib.util
import sys

import pytest
import torch

from tests.parity.gen_fixtures import ensure_accelerate


@pytest.mark.skipif(importlib.util.find_spec("accelerate") is not None, reason="accelerate installed: no shim")
def test_ensure_accelerate_is_reentrant():
    """The default generation path builds the DiT several times in one process (fixtures, floors, calls, blocks)."""
    try:
        ensure_accelerate(torch)
        ensure_accelerate(torch)
        with sys.modules["accelerate"].init_empty_weights():
            layer = torch.nn.Linear(4, 4)
        assert layer.weight.device.type == "meta" and torch.nn.Linear(4, 4).weight.device.type == "cpu"
    finally:
        sys.modules.pop("accelerate", None)


def test_scene_specs():
    from tests.parity.gen_fixtures import SCENE_SPECS
    assert set(SCENE_SPECS) == {"ancient-egypt", "castle", "racing"}
    assert (SCENE_SPECS["racing"].sink_mode, SCENE_SPECS["racing"].sink_rope_mode) == ("first_frame", "absolute")
    assert SCENE_SPECS["racing"].src_dir == "driving_game"
    assert (SCENE_SPECS["castle"].sink_mode, SCENE_SPECS["castle"].sink_rope_mode) == ("kv_only", "fixed")
    assert SCENE_SPECS["castle"].src_dir == "fps-castle" and SCENE_SPECS["castle"].fixture_dir == "castle_quality"
    assert SCENE_SPECS["ancient-egypt"].fixture_dir == "egypt_quality"
    assert SCENE_SPECS["ancient-egypt"].checkpoint.parts[-2:] == ("fps-egypt", "model.safetensors")


def test_scene_out_dir():
    from tests.conftest import fixtures_dir
    from tests.parity.gen_fixtures import parse_args, scene_out_dir
    assert scene_out_dir(parse_args(["--scene", "castle", "--calls-only"])) == fixtures_dir() / "castle_quality"
    assert scene_out_dir(parse_args([])) == fixtures_dir() / "egypt_quality"


def test_first_denoising_call():
    from tests.parity.gen_fixtures import first_denoising_call
    names = [("calls_chunk0", 1), ("calls_chunk1", 3)]
    assert first_denoising_call(names, {"calls_chunk0": [True], "calls_chunk1": [False, False, True]}) == ("calls_chunk1", 0)
    names = [("calls_chunk0", 3), ("calls_chunk1", 3)]
    assert first_denoising_call(names, {"calls_chunk0": [False, False, True], "calls_chunk1": [False] * 3}) == ("calls_chunk0", 0)


def test_expected_denoising_calls_per_chunk():
    from tests.parity.gen_fixtures import SCENE_SPECS, STEPS, n_denoising_calls
    assert n_denoising_calls(SCENE_SPECS["racing"], 0) == 0
    assert n_denoising_calls(SCENE_SPECS["racing"], 1) == len(STEPS)
    assert n_denoising_calls(SCENE_SPECS["castle"], 0) == len(STEPS)
