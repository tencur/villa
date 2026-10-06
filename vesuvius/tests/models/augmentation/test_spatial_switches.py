"""--no-spatial and rotation_axes must reach the training augmentation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms


def _spatial_names(pipeline):
    names = []
    for transform in pipeline.transforms:
        inner = getattr(transform, "transform", transform)
        options = getattr(inner, "list_of_transforms", [inner])
        names.extend(type(option).__name__ for option in options)
    return [n for n in names if n in ("Rot90Transform", "TransposeAxesTransform")]


def _z_profile_kept(pipeline, draws=40):
    """Apply the pipeline to a z-asymmetric label and check per-slice counts stay the same."""
    label = torch.zeros(1, 16, 16, 16)
    label[0, :4] = 1  # a slab at the top of the depth axis
    label[0, :, :8, :2] = 1  # and something asymmetric in y-x
    profile = label[0].sum(dim=(1, 2))
    np.random.seed(0)
    for _ in range(draws):
        out = pipeline(image=torch.rand(1, 16, 16, 16), label=label.clone())["label"][0]
        got = out.sum(dim=(1, 2))
        if not (torch.equal(got, profile) or torch.equal(got, profile.flip(0))):
            return False
    return True


def test_rotation_about_z_only_turns_the_y_x_plane():
    pipeline = create_training_transforms((16, 16, 16), allowed_rotation_axes=[0])
    assert set(_spatial_names(pipeline)) == {"Rot90Transform", "TransposeAxesTransform"}
    assert _z_profile_kept(pipeline)


def test_without_a_restriction_rotations_move_the_z_axis():
    assert not _z_profile_kept(create_training_transforms((16, 16, 16)))


def test_no_rotation_axes_means_no_rotation_or_transpose():
    assert _spatial_names(create_training_transforms((16, 16, 16), allowed_rotation_axes=[])) == []


def _dataset(tmp_path: Path, **mgr_fields):
    import zarr
    from vesuvius.models.datasets.zarr_dataset import ZarrDataset

    for sub, name in (("images", "vol.zarr"), ("labels", "vol_ink.zarr")):
        array = zarr.open(str(tmp_path / sub / name), mode="w", shape=(16, 16, 16), dtype="u1")
        array[:] = 1
    mgr = SimpleNamespace(
        data_path=tmp_path, train_patch_size=(16, 16, 16), targets={"ink": {}},
        skip_patch_validation=True, normalization_scheme="none", intensity_properties={},
        **mgr_fields,
    )
    return ZarrDataset(mgr, is_training=True)


def test_dataset_honours_no_spatial_as_the_config_and_cli_set_it(tmp_path):
    assert _spatial_names(_dataset(tmp_path, no_spatial=True).transforms) == []


def test_dataset_passes_rotation_axes_on(tmp_path):
    dataset = _dataset(tmp_path, allowed_rotation_axes=(0,))
    assert _z_profile_kept(dataset.transforms)
