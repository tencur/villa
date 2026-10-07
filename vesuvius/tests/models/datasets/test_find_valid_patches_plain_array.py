"""A label stored as a bare zarr array (no pyramid) must be scanned at full resolution.

With the default valid_patch_find_resolution=1 the resolver used to hand the bare array back as if it were the
level-1 pyramid array: the scan then used half-size footprints and doubled every cached position.
"""

from __future__ import annotations

import numpy as np
import zarr

from vesuvius.models.datasets.find_valid_patches import find_valid_patches


def test_bare_array_positions_stay_inside_the_volume(tmp_path):
    labels = np.zeros((64, 64, 64), np.uint8)
    labels[8:56, 8:56, 8:56] = 1
    arr = zarr.open(str(tmp_path / "v_ink.zarr"), mode="w", shape=labels.shape, chunks=(32, 32, 32), dtype="u1")
    arr[:] = labels

    result = find_valid_patches(
        label_arrays=[zarr.open(str(tmp_path / "v_ink.zarr"), mode="r")],
        label_names=["v"],
        patch_size=(32, 32, 32),
        bbox_threshold=0.3,   # a 32^3 patch at the corner holds a 24^3 labelled block: bbox share 0.42
        label_threshold=0.1,
        valid_patch_find_resolution=1,
    )
    positions = sorted(tuple(int(v) for v in p["start_pos"]) for p in result["fg_patches"])
    assert positions, "the labelled block must yield foreground patches"
    for pos in positions:
        assert all(0 <= v <= 32 for v in pos), f"patch start {pos} lies outside a 64^3 volume with 32^3 patches"
    assert (0, 0, 0) in positions and (32, 32, 32) in positions
