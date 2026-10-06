"""The single-point tracer must keep -1 at empty vertices when it scales a traced patch for saving."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

import vesuvius

TRACE_DIR = Path(vesuvius.__file__).parent / "neural_tracing" / "heatmap_single_point"


@pytest.mark.parametrize("as_tensor", [False, True])
def test_empty_vertices_stay_minus_one(as_tensor):
    tifxyz = importlib.import_module("vesuvius.neural_tracing.heatmap_single_point.tifxyz")
    patch = np.full((4, 4, 3), -1.0)
    patch[1:3, 1:3] = [10.0, 20.0, 30.0]
    if as_tensor:
        patch = torch.from_numpy(patch)
    out = tifxyz.scale_patch_for_save(patch, 2)
    assert np.all(out[0, 0] == -1)
    assert np.allclose(out[1, 1], [40.0, 80.0, 120.0])


def test_every_save_in_trace_uses_the_guard():
    # trace.py itself imports GPU-only modules, so check its source text
    source = (TRACE_DIR / "trace.py").read_text()
    assert "patch_zyxs * 2 ** volume_scale" not in source
    assert source.count("scale_patch_for_save(") == 3
