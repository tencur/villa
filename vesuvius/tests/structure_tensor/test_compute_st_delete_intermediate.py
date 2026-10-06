"""vesuvius.compute_st --delete-intermediate must keep the eigenanalysis results."""

from __future__ import annotations

import sys

import numpy as np
import zarr

from vesuvius.structure_tensor import run_create_st


def test_delete_intermediate_removes_only_the_structure_tensor(tmp_path, monkeypatch):
    output = tmp_path / "st"
    root = tmp_path / "st.zarr"

    def fake_part(args, part_id, gpu_id, output_path):
        group = zarr.open_group(output_path, mode="a")
        group.create_array("structure_tensor", shape=(6, 4, 4, 4), dtype="f4")
        return True

    def fake_eigen(zarr_path, **kwargs):
        group = zarr.open_group(zarr_path, mode="a")
        for name in ("first_component", "second_component", "normal", "confidence"):
            group.create_array(name, shape=(4, 4, 4), dtype="u1")
        return True

    monkeypatch.setattr(run_create_st, "run_structure_tensor_part", fake_part)
    monkeypatch.setattr(run_create_st, "run_eigenanalysis", fake_eigen)
    monkeypatch.setattr(
        sys, "argv",
        ["vesuvius.compute_st", "--input_dir", str(tmp_path / "in.zarr"), "--output_dir", str(output),
         "--delete-intermediate"],
    )
    monkeypatch.setattr(run_create_st, "select_gpus", lambda gpus: [])  # the single-device path
    zarr.open(str(tmp_path / "in.zarr"), mode="w", shape=(4, 4, 4), dtype="u1")

    assert run_create_st.main() == 0

    assert root.is_dir()
    kept = sorted(name for name, _ in zarr.open_group(str(root), mode="r").arrays())
    assert kept == ["confidence", "first_component", "normal", "second_component"]
