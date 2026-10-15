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
