"""edt-dilate chunk by chunk must equal a dilation of the whole volume."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import zarr
from scipy.ndimage import distance_transform_edt

from vesuvius.data.utils import create_zarr_array, open_zarr_group


def _dilate(tmp_path, labels, distance, *extra):
    group = open_zarr_group(str(tmp_path / "in.zarr"), mode="w")
    create_zarr_array(group, "0", data=labels, chunks=(8, 8, 8))
    result = subprocess.run(
        [sys.executable, "-m", "vesuvius.image_proc.run.zarr_tasks", str(tmp_path / "in.zarr"),
         str(tmp_path / "out.zarr"), "--task", "edt-dilate", "--distance", str(distance),
         "--chunk-size", "8,8,8", "--num-workers", "1", *extra],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return np.asarray(zarr.open(str(tmp_path / "out.zarr"), mode="r")["0"][:]) > 0


def test_chunked_dilation_equals_the_whole_volume_dilation(tmp_path):
    labels = np.zeros((24, 24, 24), dtype=np.uint8)
    labels[7:9, 7:9, 7:9] = 1      # straddles chunk faces at 8
    labels[20, 3, 12] = 1          # one voxel near the volume edge
    expected = distance_transform_edt(labels == 0) <= 3

    got = _dilate(tmp_path, labels, 3)

    np.testing.assert_array_equal(got, expected)
    assert not got[0, 0, 0] and not got[23, 23, 23]  # no shells on faces or corners


def test_black_border_false_treats_the_outside_as_foreground(tmp_path):
    labels = np.zeros((16, 16, 16), dtype=np.uint8)
    labels[8, 8, 8] = 1

    got = _dilate(tmp_path, labels, 1, "--black-border", "false")

    assert got[0, 8, 8] and got[8, 8, 8] and not got[4, 4, 4]
