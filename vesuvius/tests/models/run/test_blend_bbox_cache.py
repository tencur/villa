"""Bounding boxes cached by one blend must not decide the parts of a later inference run."""

from __future__ import annotations

import os
import time

import numpy as np
import zarr

from vesuvius.data.utils import open_zarr
from vesuvius.models.run.blending import merge_inference_outputs

PATCH = (8, 8, 8)
VOLUME = (16, 8, 8)


def _write_part(parent, part_id, z_positions):
    """One part as vesuvius.predict writes it (with the bbox it records), possibly with no patches."""
    n = len(z_positions)
    logits = open_zarr(path=os.path.join(parent, f"logits_part_{part_id}.zarr"), mode="w",
                       shape=(max(n, 1), 2, *PATCH), chunks=(1, 2, *PATCH), dtype=np.float16,
                       compressor=None, write_empty_chunks=False)
    for i in range(n):
        logits[i, 0] = -3.0
        logits[i, 1] = 3.0
    logits.attrs.update({"patch_size": list(PATCH), "original_volume_shape": list(VOLUME),
                         "part_id": part_id, "num_parts": 2})
    coords = open_zarr(path=os.path.join(parent, f"coordinates_part_{part_id}.zarr"), mode="w",
                       shape=(n, 3), chunks=(max(n, 1), 3), dtype=np.int32, compressor=None,
                       write_empty_chunks=False)
    if n:
        coords[:] = [(z, 0, 0) for z in z_positions]
        coords.attrs["bbox"] = {"z_min": min(z_positions), "z_max": max(z_positions) + 8,
                                "y_min": 0, "y_max": 8, "x_min": 0, "x_max": 8}


def _blend(parent, output):
    merge_inference_outputs(str(parent), str(output), chunk_size=PATCH, num_workers=1,
                            compression_level=0, verbose=False)
    return np.asarray(zarr.open(str(output), mode="r")[:])


def test_part_without_patches_in_an_earlier_run_is_blended_in_a_later_run(tmp_path):
    parent = tmp_path / "logits"
    parent.mkdir()
    # Run 1 (a small --bbox): only part 0 gets a patch.
    _write_part(parent, 0, [0])
    _write_part(parent, 1, [])
    first = _blend(parent, tmp_path / "first.zarr")
    assert (parent / ".bbox_cache.json").is_file()
    assert first[1, 4, 4, 4] > 2.9 and not first[:, 8:].any()

    time.sleep(0.05)
    # Run 2 into the same folder (a larger --bbox): part 1 now has a patch.
    _write_part(parent, 0, [0])
    _write_part(parent, 1, [8])
    now = time.time() + 2
    for part in (0, 1):
        os.utime(parent / f"coordinates_part_{part}.zarr" / ".zarray", (now, now))
    second = _blend(parent, tmp_path / "second.zarr")

    assert second[1, 12, 4, 4] > 2.9  # part 1's patch is in the merged result


def test_cached_boxes_are_reused_while_the_parts_are_unchanged(tmp_path, monkeypatch):
    parent = tmp_path / "logits"
    parent.mkdir()
    _write_part(parent, 0, [0])
    _write_part(parent, 1, [8])
    first = _blend(parent, tmp_path / "first.zarr")
    later = time.time() + 5
    os.utime(parent / ".bbox_cache.json", (later, later))

    import vesuvius.models.run.blending as blending
    calls = []
    original = blending.SpatialPatchGrid._save_aggregate_bbox_cache
    monkeypatch.setattr(blending.SpatialPatchGrid, "_save_aggregate_bbox_cache",
                        lambda self, d: calls.append(d) or original(self, d))
    second = _blend(parent, tmp_path / "second.zarr")

    assert calls == []  # the cache was used, not rebuilt
    np.testing.assert_array_equal(first, second)
