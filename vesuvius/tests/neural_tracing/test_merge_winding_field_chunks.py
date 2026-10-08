"""merge_winding_field: level-0/1 tiles that share a 128-chunk must not erase each other's planes.

make_band aligns the band origin to 32, so z_lo >> level is exact for every pyramid level, but the level-0 and
level-1 tiles (512 / 256 voxels in z) then start inside chunks and two consecutive tiles write the same chunk.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import zarr

from vesuvius.neural_tracing.winding_models import merge_winding_field as m


@pytest.mark.parametrize("z_lo", [7360, 7424])   # documented PHercParis4 run (7360 % 128 = 64) and a chunk-aligned control
def test_band_planes_survive_tiled_levels(tmp_path, z_lo):
    band = m.BandGeometry((z_lo + 1024, 64, 64), z_lo, z_lo + 1024)
    writer = m.OmeZarrPyramidWriter(str(tmp_path / "wf.zarr"), band, "src", {}, 1)
    field4 = torch.ones(band.band_shape(4))
    support8 = torch.ones(band.band_shape(8), dtype=torch.bool)
    m.write_pyramid(field4, support8, band, writer, m.Config(patches_dir="", mask_dilate=0), torch.device("cpu"))
    writer.finalize({})
    for level, scale in ((0, 1), (1, 2), (2, 4)):
        arr = zarr.open(str(tmp_path / "wf.zarr" / str(level)), mode="r")
        inside = np.asarray(arr[z_lo // scale:(z_lo + 1024) // scale])
        zero_planes = int((inside == 0).all(axis=(1, 2)).sum() + ((inside == 0).any(axis=(1, 2)) & ~(inside == 0).all(axis=(1, 2))).sum())
        assert zero_planes == 0, f"level {level}: {zero_planes} zeroed planes inside the band"
