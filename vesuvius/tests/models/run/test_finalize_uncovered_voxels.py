"""Voxels that no patch covered must not become foreground when finalized."""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr

from vesuvius.data.utils import open_zarr
from vesuvius.models.run.finalize_outputs import FinalizeConfig, apply_finalization, finalize_logits


def _chunk_half_covered(channels: int) -> np.ndarray:
    """A blended chunk as blend_logits writes it: one half predicted background, one half no patch."""
    logits = np.zeros((channels, 2, 2, 4), dtype=np.float32)
    if channels == 2:
        logits[0, ..., :2] = 3.0
        logits[1, ..., :2] = -3.0
    else:
        logits[0, ..., :2] = -3.0
    return logits


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("threshold", [0.2, 0.3, 0.5, None])
def test_voxels_marked_no_prediction_are_not_foreground(channels, threshold):
    logits = _chunk_half_covered(channels)
    output, is_empty = apply_finalization(
        logits,
        channels,
        FinalizeConfig(mode="binary", threshold=threshold),
        no_prediction=np.all(logits == 0, axis=0),
    )
    assert not is_empty
    assert not output[..., 2:].any()
    assert (output[..., :2] < 128).all()


def test_finalize_command_writes_no_foreground_where_no_patch_was(tmp_path):
    merged = str(tmp_path / "merged.zarr")
    store = open_zarr(path=merged, mode="w", shape=(2, 4, 4, 8), chunks=(2, 4, 4, 8),
                      dtype=np.float16, compressor=None)
    store[:] = _chunk_half_covered(2).repeat(2, axis=1).repeat(2, axis=2).repeat(2, axis=3)
    output = str(tmp_path / "final.zarr")

    finalize_logits(merged, output, mode="binary", threshold=0.3, num_workers=1, verbose=False)

    final = zarr.open(output, mode="r")
    final = final["0"] if isinstance(final, zarr.Group) else final
    assert not np.asarray(final[:]).any()
