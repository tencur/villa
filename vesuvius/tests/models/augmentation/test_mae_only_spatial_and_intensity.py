"""The MAE trainers set mgr.only_spatial_and_intensity; the training datasets must pass it on.

MAE reconstructs its own augmented input, so the noise, blur, sharpening, low-resolution, smear and
local transforms that the flag removes would otherwise become reconstruction targets.
"""

from __future__ import annotations

from types import SimpleNamespace

from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.models.augmentation.transforms.utils.perf import collect_augmentation_names
from vesuvius.models.datasets.zarr_dataset import ZarrDataset

EXCLUDED = {"GaussianNoiseTransform", "GaussianBlurTransform", "SharpeningTransform",
            "SimulateLowResolutionTransform", "SmearTransform"}


def _dataset_pipeline(flag: bool):
    ns = SimpleNamespace(
        mgr=SimpleNamespace(only_spatial_and_intensity=flag),
        is_training=True,
        patch_size=(32, 32, 32),
        _profile_augmentations=False,
        _get_skeleton_targets=lambda: ([], {}),
    )
    ZarrDataset._initialize_transforms(ns)
    return set(collect_augmentation_names(ns.transforms))


def test_flag_reaches_zarr_dataset_pipeline() -> None:
    expected = set(collect_augmentation_names(
        create_training_transforms(patch_size=(32, 32, 32), only_spatial_and_intensity=True)))
    names = _dataset_pipeline(True)
    assert names == expected
    assert not (names & EXCLUDED)


def test_default_pipeline_unchanged() -> None:
    names = _dataset_pipeline(False)
    assert names == set(collect_augmentation_names(create_training_transforms(patch_size=(32, 32, 32))))
    assert names & EXCLUDED
