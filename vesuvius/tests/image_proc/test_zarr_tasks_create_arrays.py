"""edt-dilate, merge, resize and remap must create their outputs under zarr 2 and zarr 3 alike."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr

from vesuvius.data.utils import create_zarr_array, open_zarr_group


def _group(path: Path, levels: list[np.ndarray]) -> str:
    group = open_zarr_group(str(path), mode="w")
    for level, data in enumerate(levels):
        create_zarr_array(group, str(level), data=data, chunks=(8, 8, 8))
    return str(path)


def _run(*argv: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "vesuvius.image_proc.run.zarr_tasks", *argv, "--num-workers", "1"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def _labels() -> np.ndarray:
    labels = np.zeros((16, 16, 16), dtype=np.uint8)
    labels[6:10, 6:10, 6:10] = 255
    return labels


def test_edt_dilate_creates_its_output(tmp_path):
    source = _group(tmp_path / "in.zarr", [_labels()])
    _run(source, str(tmp_path / "out.zarr"), "--task", "edt-dilate", "--distance", "1")
    out = np.asarray(zarr.open(str(tmp_path / "out.zarr"), mode="r")["0"][:])
    assert out[8, 8, 8] == 1 and out[5, 8, 8] == 1 and out[3, 3, 3] == 0


def test_merge_creates_its_output(tmp_path):
    first = _labels()
    second = np.zeros_like(first); second[0:2] = 7
    a = _group(tmp_path / "a.zarr", [first]); b = _group(tmp_path / "b.zarr", [second])
    _run(a, str(tmp_path / "out.zarr"), "--task", "merge", "--input2", b, "--num-levels", "1")
    out = np.asarray(zarr.open(str(tmp_path / "out.zarr"), mode="r")["0"][:])
    np.testing.assert_array_equal(out, np.maximum(first, second))


def test_resize_creates_its_output(tmp_path):
    source = _group(tmp_path / "in.zarr", [_labels()])
    reference = _group(tmp_path / "ref.zarr", [np.zeros((8, 8, 8), dtype=np.uint8)])
    _run(source, str(tmp_path / "out.zarr"), "--task", "resize", "--reference", reference,
         "--interpolation", "nearest")
    out = np.asarray(zarr.open(str(tmp_path / "out.zarr"), mode="r")["0"][:])
    assert out.shape == (8, 8, 8) and out[4, 4, 4] == 255 and out[0, 0, 0] == 0


def test_remap_creates_its_output(tmp_path):
    source = _group(tmp_path / "in.zarr", [_labels()])
    _run(source, str(tmp_path / "out.zarr"), "--task", "remap", "--remap", "255:1")
    out = np.asarray(zarr.open(str(tmp_path / "out.zarr"), mode="r")["0"][:])
    np.testing.assert_array_equal(out, np.where(_labels() == 255, 1, 0))
