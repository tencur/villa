"""The cached listing of written patches must not outlive an inference run still in progress."""

from __future__ import annotations

import os
import warnings

import numpy as np
import zarr

from vesuvius.data.utils import open_zarr
from vesuvius.models.run.blending import merge_inference_outputs

PATCH = (8, 8, 8)
VOLUME = (16, 8, 8)
CACHE = ".non_empty_patch_idxs.json"


def _create_part(parent):
    """A one-part run with two patch slots, as vesuvius.predict lays it out; nothing written yet."""
    logits = open_zarr(
        path=os.path.join(parent, "logits_part_0.zarr"),
        mode="w",
        shape=(2, 2, *PATCH),
        chunks=(1, 2, *PATCH),
        dtype=np.float16,
        compressor=None,
        write_empty_chunks=False,
    )
    logits.attrs["patch_size"] = list(PATCH)
    logits.attrs["original_volume_shape"] = list(VOLUME)
    logits.attrs["part_id"] = 0
    logits.attrs["num_parts"] = 1
    coordinates = open_zarr(
        path=os.path.join(parent, "coordinates_part_0.zarr"),
        mode="w",
        shape=(2, 3),
        chunks=(2, 3),
        dtype=np.int32,
        compressor=None,
        write_empty_chunks=False,
    )
    coordinates[:] = [(0, 0, 0), (8, 0, 0)]
    return logits


def _write_patch(logits, index: int) -> None:
    logits[index, 0] = -3.0
    logits[index, 1] = 3.0


def _blend(parent, tmp_path, name: str):
    output = str(tmp_path / f"{name}.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merge_inference_outputs(
            str(parent), output, chunk_size=PATCH, num_workers=1, compression_level=0, verbose=False
        )
    return np.asarray(zarr.open(output, mode="r")[:])


def test_blend_run_during_inference_does_not_hide_patches_written_later(tmp_path):
    parent = tmp_path / "logits"
    parent.mkdir()
    logits = _create_part(parent)
    _write_patch(logits, 0)

    early = _blend(parent, tmp_path, "early")  # inference has written one of two patches
    assert early[1, 4, 4, 4] > 2.9 and not early[:, 8:].any()

    _write_patch(logits, 1)  # inference finishes
    logits.attrs["inference_complete"] = True
    final = _blend(parent, tmp_path, "final")

    assert final[1, 4, 4, 4] > 2.9
    assert final[1, 12, 4, 4] > 2.9  # the second patch is in the merged result


def test_listing_of_a_part_not_marked_complete_is_not_cached(tmp_path):
    parent = tmp_path / "logits"
    parent.mkdir()
    logits = _create_part(parent)
    _write_patch(logits, 0)

    _blend(parent, tmp_path, "early")

    assert not (parent / "logits_part_0.zarr" / CACHE).exists()


def test_listing_of_a_complete_part_is_cached(tmp_path):
    parent = tmp_path / "logits"
    parent.mkdir()
    logits = _create_part(parent)
    _write_patch(logits, 0)
    _write_patch(logits, 1)
    logits.attrs["inference_complete"] = True

    first = _blend(parent, tmp_path, "first")
    assert (parent / "logits_part_0.zarr" / CACHE).is_file()
    second = _blend(parent, tmp_path, "second")

    np.testing.assert_array_equal(first, second)
    assert second[1, 12, 4, 4] > 2.9
