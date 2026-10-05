"""blend_logits must not merge an incomplete or mixed set of inference parts."""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr

from vesuvius.data.utils import open_zarr
from vesuvius.models.run.blending import merge_inference_outputs

PATCH = (8, 8, 8)
VOLUME = (16, 8, 8)


def _write_part(parent, part_id: int, num_parts: int, z: int) -> None:
    """One part with a single patch, laid out as vesuvius.predict writes it."""
    logits = open_zarr(
        path=os.path.join(parent, f"logits_part_{part_id}.zarr"),
        mode="w",
        shape=(1, 2, *PATCH),
        chunks=(1, 2, *PATCH),
        dtype=np.float16,
        compressor=None,
        write_empty_chunks=False,
    )
    logits[0, 0] = -3.0
    logits[0, 1] = 3.0
    logits.attrs["patch_size"] = list(PATCH)
    logits.attrs["original_volume_shape"] = list(VOLUME)
    logits.attrs["part_id"] = part_id
    logits.attrs["num_parts"] = num_parts
    coordinates = open_zarr(
        path=os.path.join(parent, f"coordinates_part_{part_id}.zarr"),
        mode="w",
        shape=(1, 3),
        chunks=(1, 3),
        dtype=np.int32,
        compressor=None,
        write_empty_chunks=False,
    )
    coordinates[0] = (z, 0, 0)


def _blend(parent, tmp_path):
    output = str(tmp_path / "merged.zarr")
    merge_inference_outputs(
        str(parent), output, chunk_size=PATCH, num_workers=1, compression_level=0, verbose=False
    )
    return np.asarray(zarr.open(output, mode="r")[:])


def test_blend_refuses_a_run_with_a_missing_part(tmp_path):
    parent = tmp_path / "logits"
    parent.mkdir()
    _write_part(parent, 0, num_parts=2, z=0)  # the job for part 1 never finished

    with pytest.raises(FileNotFoundError, match=r"split into 2 parts \(missing: \[1\]"):
        _blend(parent, tmp_path)


def test_blend_refuses_parts_left_by_a_run_with_a_different_split(tmp_path):
    parent = tmp_path / "logits"
    parent.mkdir()
    _write_part(parent, 0, num_parts=1, z=0)
    _write_part(parent, 1, num_parts=2, z=8)  # left over from an earlier two-part run

    with pytest.raises(FileNotFoundError, match=r"not part of that run: \[1\]"):
        _blend(parent, tmp_path)


def test_blend_merges_a_complete_run(tmp_path):
    parent = tmp_path / "logits"
    parent.mkdir()
    _write_part(parent, 0, num_parts=2, z=0)
    _write_part(parent, 1, num_parts=2, z=8)

    merged = _blend(parent, tmp_path)

    assert merged.shape == (2, *VOLUME)
    # Both parts' patches are in the result (centre voxels; the Gaussian weighting
    # attenuates the corners of a patch nothing else overlaps).
    for z_centre in (4, 12):
        assert merged[1, z_centre, 4, 4] == pytest.approx(3.0, abs=1e-2)
        assert merged[0, z_centre, 4, 4] == pytest.approx(-3.0, abs=1e-2)
