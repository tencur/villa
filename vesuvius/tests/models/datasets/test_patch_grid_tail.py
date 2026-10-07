"""Patch grids must reach the end of every axis: voxels past the last full patch were never trained or validated."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from vesuvius.models.datasets.find_valid_patches import find_valid_patches, grid_starts


def test_grid_starts_adds_an_end_aligned_patch():
    assert grid_starts(0, 320, 192, 192) == [0, 128]
    assert grid_starts(0, 384, 192, 192) == [0, 192]          # exact multiple: unchanged
    assert grid_starts(0, 100, 192, 192) == []                # shorter than a patch (find_valid_patches)
    assert grid_starts(0, 100, 192, 192, allow_short=True) == [0]
    assert grid_starts(10, 330, 192, 192) == [10, 138]


def test_find_valid_patches_covers_labels_in_the_tail(tmp_path):
    labels = np.zeros((96, 96, 96), np.uint8)
    labels[70:90, 70:90, 70:90] = 1                         # only in the tail strip of a 64-voxel grid
    root = zarr.open_group(str(tmp_path / "v_ink.zarr"), mode="w")
    root.create_array("0", data=labels, chunks=(32, 32, 32))
    result = find_valid_patches(
        label_arrays=[zarr.open_group(str(tmp_path / "v_ink.zarr"), mode="r")],
        label_names=["v"],
        patch_size=(64, 64, 64),
        bbox_threshold=0.0,
        label_threshold=0.0001,
        valid_patch_find_resolution=0,
    )
    starts = sorted(tuple(int(v) for v in p["start_pos"]) for p in result["fg_patches"])
    assert (32, 32, 32) in starts, f"labelled block at 70..90 must be inside some patch, got {starts}"


def test_stride_groups_keep_irregular_tail_separate():
    from vesuvius.models.datasets.find_valid_patches import _stride_groups
    assert _stride_groups([0, 32, 64, 68], 2, 32) == [[0, 32], [64], [68]]
    assert _stride_groups([0, 32, 64, 96], 2, 32) == [[0, 32], [64, 96]]


def test_tail_start_sharing_a_chunk_reads_only_real_labels(tmp_path):
    # chunk = 2 x patch, 100 voxels per axis: starts 0, 32, 64, 68 -> the tail (68) used to share a block with 64,
    # and the strided view read past the block. An all-zero label must give no foreground patch at all.
    labels = np.zeros((100, 100, 100), np.uint8)
    root = zarr.open_group(str(tmp_path / "z_ink.zarr"), mode="w")
    root.create_array("0", data=labels, chunks=(64, 64, 64))
    empty = find_valid_patches(
        label_arrays=[zarr.open_group(str(tmp_path / "z_ink.zarr"), mode="r")], label_names=["z"],
        patch_size=(32, 32, 32), bbox_threshold=0.0, label_threshold=0.0001, valid_patch_find_resolution=0)
    assert empty["fg_patches"] == []

    labels[97:100, 97:100, 97:100] = 1                       # past 96: only the tail patch [68, 100) sees this
    root2 = zarr.open_group(str(tmp_path / "t_ink.zarr"), mode="w")
    root2.create_array("0", data=labels, chunks=(64, 64, 64))
    found = find_valid_patches(
        label_arrays=[zarr.open_group(str(tmp_path / "t_ink.zarr"), mode="r")], label_names=["t"],
        patch_size=(32, 32, 32), bbox_threshold=0.0, label_threshold=0.0001, valid_patch_find_resolution=0)
    starts = sorted(tuple(int(v) for v in p["start_pos"]) for p in found["fg_patches"])
    assert starts == [(68, 68, 68)], starts


@pytest.mark.parametrize("level1_side", [48, 49], ids=["pyramid_rounded_down", "pyramid_rounded_up"])
def test_downsampled_tail_patch_ends_exactly_at_the_volume_end(tmp_path, level1_side):
    # 97^3 at full resolution: level 1 is 48^3 or 49^3 depending on how the pyramid rounds. Scaling the
    # level-1 tail start by 2 would end the last patch at 96 (missing voxel 96) or at 98 (past the volume).
    labels = np.ones((97, 97, 97), np.uint8)
    root = zarr.open_group(str(tmp_path / "p_ink.zarr"), mode="w")
    root.create_array("0", data=labels, chunks=(32, 32, 32))
    root.create_array("1", data=np.ones((level1_side,) * 3, np.uint8), chunks=(16, 16, 16))
    result = find_valid_patches(
        label_arrays=[zarr.open_group(str(tmp_path / "p_ink.zarr"), mode="r")], label_names=["p"],
        patch_size=(32, 32, 32), bbox_threshold=0.0, label_threshold=0.0001, valid_patch_find_resolution=1)
    starts = np.array([p["start_pos"] for p in result["fg_patches"]])
    assert starts.min() >= 0
    assert (starts.max(axis=0) + 32 == 97).all(), starts.max(axis=0)
