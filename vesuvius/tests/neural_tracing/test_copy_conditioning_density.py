"""infer_rowcol_triplet_wraps must give the model the dense conditioning sheet it was trained on.

Training (dataset_rowcol_cond) upsamples the stored lattice by 1/scale with Catmull-Rom before
voxelizing the conditioning channel. Inference rasterized only the lattice vertices and the lines
between them, so a lattice stored at scale 0.05 and run at --volume-scale 1 (10 voxels per step)
reached the model as a wireframe with 10x10-voxel holes.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

from vesuvius.neural_tracing.datasets.common import _upsample_world_surface, voxelize_surface_grid

CROP = (32, 64, 64)
STEP = 10


@pytest.fixture
def infer(monkeypatch):
    if "vc" not in sys.modules:
        monkeypatch.setitem(sys.modules, "vc", types.ModuleType("vc"))
    return importlib.import_module("vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps")


def _tilted_lattice(rows=6, cols=6):
    """A gently tilted plane sampled every STEP voxels in y and x: local zyx and uv per vertex."""
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    y = 4.0 + STEP * r
    x = 4.0 + STEP * c
    z = 10.0 + 0.15 * y + 0.05 * x
    local = np.stack([z, y, x], axis=-1).reshape(-1, 3).astype(np.float32)
    uv = np.stack([r, c], axis=-1).reshape(-1, 2)
    return local, uv, (rows, cols)


def _training_style(local, shape):
    grid = local.reshape(shape + (3,))
    x_up, y_up, z_up = _upsample_world_surface(grid[..., 2], grid[..., 1], grid[..., 0], 1 / STEP, 1 / STEP)
    return voxelize_surface_grid(np.stack([z_up, y_up, x_up], axis=-1).astype(np.float32), CROP) > 0


def test_conditioning_matches_the_training_sheet(infer) -> None:
    local, uv, shape = _tilted_lattice()
    train = _training_style(local, shape)
    dense = infer._voxelize_local_surface_from_uv_points(local, uv, CROP, scale_rc=(1 / STEP, 1 / STEP)) > 0
    assert train.sum() > 1500
    # every voxel of the training-style sheet is present at inference
    assert np.all(dense[train])
    # and nothing far from it is added
    assert dense.sum() <= 1.05 * train.sum()


def test_without_scale_only_the_wireframe_is_drawn(infer) -> None:
    local, uv, shape = _tilted_lattice()
    train = _training_style(local, shape)
    wire = infer._voxelize_local_surface_from_uv_points(local, uv, CROP) > 0
    assert wire.sum() < 0.3 * train.sum()


def test_invalid_vertex_leaves_a_hole_and_keeps_the_rest(infer) -> None:
    local, uv, shape = _tilted_lattice()
    keep = ~((uv[:, 0] == 2) & (uv[:, 1] == 2))
    dense = infer._voxelize_local_surface_from_uv_points(
        local[keep], uv[keep], CROP, scale_rc=(1 / STEP, 1 / STEP)
    ) > 0
    full = infer._voxelize_local_surface_from_uv_points(local, uv, CROP, scale_rc=(1 / STEP, 1 / STEP)) > 0
    # the cell around the missing vertex is not invented ...
    z, y, x = np.rint(local[~keep][0]).astype(int)
    assert not dense[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2].any()
    # ... every remaining vertex is still drawn, and most of the sheet survives
    for zz, yy, xx in np.rint(local[keep]).astype(int):
        assert dense[zz, yy, xx]
    assert dense.sum() > 0.5 * full.sum()


def test_run_passes_the_retargeted_lattice_scale(infer, monkeypatch) -> None:
    seen = {}

    def fake_inference(**kwargs):
        seen["cond_scale_rc"] = kwargs.get("cond_scale_rc")
        raise RuntimeError("stop")

    monkeypatch.setattr(infer, "_run_triplet_inference", fake_inference)
    local, uv, shape = _tilted_lattice()
    grid = local.reshape(shape + (3,))
    valid = np.ones(shape, dtype=bool)
    with pytest.raises(RuntimeError, match="stop"):
        infer._run_single_iteration(
            args=types.SimpleNamespace(verbose=False, bbox_overlap=0.0, bbox_prune=False,
                                       bbox_prune_max_remove_per_band=None, bbox_band_workers=1),
            model_state={}, crop_size=CROP, volume_arr=None, input_tifxyz_path="seg",
            out_dir=".", out_prefix="seg", retarget_factor=2.0, tifxyz_step_size=20,
            tifxyz_voxel_size_um=2.4, stored_scale_rc=(0.05, 0.05), displacement_scale=1.0,
            displacement_scale_source="default", save_scale_factor=2, iteration_index=1,
            iterations_requested=1, iterative_mode=False, iter_direction=None,
            keep_previous_wrap=False, preloaded_input=(None, grid, valid),
        )
    assert seen["cond_scale_rc"] == pytest.approx((0.1, 0.1))
