"""vesuvius.find_patches / vesuvius.train must honour ignore_label and bg_sampling_enabled.

Label layout (32^3 volume, 16^3 patches, one 32^3 storage chunk so all eight patches
are read as a single block), 2 = not annotated:
    A (0, 0, 0)    surface (1) plus annotated background (0) plus unannotated (2)  -> FG
    B (0, 0, 16)   annotated background (0) plus unannotated (2), no surface        -> BG-only
    C (0, 16, 0)   entirely unannotated (2)                                         -> neither
    D (0, 16, 16)  entirely annotated background (0)                                -> neither
    lower half     entirely unannotated (2)                                         -> neither
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.preprocessing.patches import generate_patch_caches

from tests.zarr_utils import create_v2_array

A, B = (0, 0, 0), (0, 0, 16)


def _write(path: Path, data: np.ndarray) -> None:
    array = create_v2_array(path, shape=data.shape, chunks=(32, 32, 32), dtype=data.dtype)
    array[...] = data


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    _write(tmp_path / "images" / "vol.zarr", rng.integers(1, 255, size=(32, 32, 32), dtype=np.uint8))
    label = np.full((32, 32, 32), 2, dtype=np.uint8)
    label[0:8, 0:16, 0:16] = 0
    label[0:8, 0:8, 0:8] = 1          # A: surface + background + unannotated
    label[0:8, 0:16, 16:32] = 0       # B: background + unannotated
    label[0:16, 16:32, 16:32] = 0     # D: background only
    _write(tmp_path / "labels" / "vol_ink.zarr", label)
    return tmp_path


def _config(data_path: Path, *, valid_patch_value: bool, ignore: bool = True, bg: bool = True) -> Path:
    config = data_path / "train.yaml"
    config.write_text(
        "tr_config:\n  patch_size: [16, 16, 16]\n"
        "dataset_config:\n"
        + ("  valid_patch_value: 1\n" if valid_patch_value else "")
        + "  min_labeled_ratio: 0.01\n  min_bbox_percent: 0.1\n"
        "  valid_patch_find_resolution: 0\n  normalization_scheme: none\n"
        f"  bg_sampling_enabled: {'true' if bg else 'false'}\n  bg_to_fg_ratio: 0.1\n"
        "  targets:\n    ink:\n      activation: none\n"
        + ("      ignore_label: 2\n" if ignore else "")
    )
    return config


def _dataset(config: Path, is_training: bool = True) -> ZarrDataset:
    mgr = ConfigManager(verbose=False)
    mgr.load_config(config)
    return ZarrDataset(mgr, is_training=is_training)


def test_bg_only_patch_mask_is_per_patch() -> None:
    from vesuvius.models.datasets.find_valid_patches import bg_only_patch_mask

    block = np.full((2, 2), 2, dtype=np.uint8).repeat(4, 0).repeat(4, 1)  # 8x8, four 4x4 patches
    block[0:2, 0:4] = 0                 # patch (0,0): background + unannotated
    block[0:2, 4:8] = 0
    block[0, 4] = 3                     # patch (0,1): background + unannotated + other class
    block[4:8, 0:4] = 0                 # patch (1,0): background only
    mask = bg_only_patch_mask(block, (2, 2), (4, 4), ignore_label=2)
    assert mask.tolist() == [[True, False], [False, False]]
    # With an explicit foreground value, the other class (3) is not foreground.
    mask = bg_only_patch_mask(block, (2, 2), (4, 4), ignore_label=2, valid_patch_value=1)
    assert mask.tolist() == [[True, True], [False, False]]


@pytest.mark.parametrize("valid_patch_value", [True, False], ids=["valid_patch_value", "no_valid_patch_value"])
def test_find_patches_and_trainer_use_ignore_label_and_bg_patches(dataset_dir: Path, valid_patch_value: bool) -> None:
    config = _config(dataset_dir, valid_patch_value=valid_patch_value)
    result = generate_patch_caches(config)
    assert (result.total_fg_patches, result.total_bg_patches) == (1, 1)

    dataset = _dataset(config)
    assert [tuple(p.position) for p in dataset.valid_patches] == [A, B]
    assert dataset.n_fg == 1
    assert dataset.patch_weights is not None and len(dataset.patch_weights) == len(dataset)
    assert dataset.patch_weights[0] == 1.0 and dataset.patch_weights[1] < 1e-3


def test_bg_patches_stay_out_when_bg_sampling_is_off(dataset_dir: Path) -> None:
    config = _config(dataset_dir, valid_patch_value=False, bg=False)
    result = generate_patch_caches(config)
    # BG-only patches are cached whenever the ignore label is known; the dataset decides whether to load them.
    assert (result.total_fg_patches, result.total_bg_patches) == (1, 1)
    dataset = _dataset(config)
    assert [tuple(p.position) for p in dataset.valid_patches] == [A]
    assert dataset.patch_weights is None


@pytest.mark.parametrize("bg_at_find, bg_at_train", [(True, False), (False, True)])
def test_toggling_bg_sampling_reuses_the_cache(dataset_dir: Path, bg_at_find: bool, bg_at_train: bool) -> None:
    # A cache miss would fall back to unvalidated patches, so bg_sampling_enabled must not change the cache key.
    generate_patch_caches(_config(dataset_dir, valid_patch_value=False, bg=bg_at_find))
    dataset = _dataset(_config(dataset_dir, valid_patch_value=False, bg=bg_at_train))
    expected = [A, B] if bg_at_train else [A]
    assert [tuple(p.position) for p in dataset.valid_patches] == expected


def test_validation_dataset_does_not_load_bg_patches(dataset_dir: Path) -> None:
    config = _config(dataset_dir, valid_patch_value=False)
    generate_patch_caches(config)
    dataset = _dataset(config, is_training=False)
    assert [tuple(p.position) for p in dataset.valid_patches] == [A]
    assert dataset.patch_weights is None


def test_cache_written_without_ignore_label_is_not_reused(dataset_dir: Path) -> None:
    # Without an ignore label every non-zero voxel counts as labelled.
    without = generate_patch_caches(_config(dataset_dir, valid_patch_value=False, ignore=False, bg=False))
    assert without.total_fg_patches == 7
    with_ignore = generate_patch_caches(_config(dataset_dir, valid_patch_value=False, bg=False))
    assert with_ignore.scanned_volumes == 1
    assert with_ignore.total_fg_patches == 1


def test_bg_weights_put_every_fg_patch_in_the_epoch() -> None:
    n_fg, n_bg, ratio = 50, 400, 0.1
    weights = [1.0] * n_fg + [1e-6 / n_bg] * n_bg   # what ZarrDataset sets
    n_bg_samples = int(n_fg * ratio / (1.0 - ratio))  # what the trainer draws
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        drawn = list(WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                           n_fg + n_bg_samples, replacement=False, generator=g))
        assert set(range(n_fg)) <= set(drawn)
        assert sum(i >= n_fg for i in drawn) == n_bg_samples


def test_background_patches_are_spread_through_the_epoch():
    """With FG weight 1.0 and a tiny BG weight, plain draws put every BG patch at the end of the epoch."""
    import torch
    from vesuvius.models.training.train import ShuffledWeightedRandomSampler

    n_fg, n_bg, n_bg_samples = 900, 300, 100
    weights = torch.tensor([1.0] * n_fg + [1e-6 / n_bg] * n_bg, dtype=torch.double)
    g = torch.Generator(); g.manual_seed(0)
    order = list(ShuffledWeightedRandomSampler(weights, n_fg + n_bg_samples, generator=g))
    assert len(order) == n_fg + n_bg_samples
    assert set(range(n_fg)) <= set(order), "every FG patch is still drawn once per epoch"
    first_quarter = order[: len(order) // 4]
    assert sum(1 for i in first_quarter if i >= n_fg) > 0, "BG patches must appear early, not only at the end"
