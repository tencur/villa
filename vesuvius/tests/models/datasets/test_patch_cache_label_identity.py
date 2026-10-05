"""The vesuvius.find_patches / vesuvius.train patch cache must describe the labels on disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.preprocessing.patches import generate_patch_caches

from tests.zarr_utils import create_v2_array


def _write(path: Path, data: np.ndarray) -> None:
    array = create_v2_array(path, shape=data.shape, chunks=(16, 16, 16), dtype=data.dtype)
    array[...] = data


def _config(data_path: Path, target: str) -> Path:
    config = data_path / f"train_{target}.yaml"
    config.write_text(
        "tr_config:\n  patch_size: [16, 16, 16]\n"
        "dataset_config:\n  min_labeled_ratio: 0.1\n  min_bbox_percent: 0.1\n"
        "  valid_patch_find_resolution: 0\n  normalization_scheme: none\n"
        f"  targets:\n    {target}:\n      activation: none\n"
    )
    return config


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    _write(tmp_path / "images" / "vol.zarr", rng.integers(1, 255, size=(32, 32, 32), dtype=np.uint8))
    ink = np.zeros((32, 32, 32), dtype=np.uint8)
    ink[:16] = 1  # four labelled 16^3 patches in the upper half
    _write(tmp_path / "labels" / "vol_ink.zarr", ink)
    surface = np.zeros((32, 32, 32), dtype=np.uint8)
    surface[16:, 16:, 16:] = 1  # one labelled patch, in the lower half
    _write(tmp_path / "labels" / "vol_surface.zarr", surface)
    return tmp_path


def _trainer_positions(config: Path) -> set[tuple[int, ...]]:
    mgr = ConfigManager(verbose=False)
    mgr.load_config(config)
    return {tuple(patch.position) for patch in ZarrDataset(mgr).valid_patches}


def test_patch_cache_is_reused_while_labels_are_unchanged(dataset_dir: Path) -> None:
    config = _config(dataset_dir, "ink")

    first = generate_patch_caches(config)
    second = generate_patch_caches(config)

    assert (first.written_caches, first.total_fg_patches) == (1, 4)
    assert (second.written_caches, second.skipped_volumes, second.total_fg_patches) == (0, 1, 4)
    assert len(_trainer_positions(config)) == 4


def test_patch_cache_is_rebuilt_after_a_label_is_edited_in_place(dataset_dir: Path) -> None:
    config = _config(dataset_dir, "ink")
    assert generate_patch_caches(config).total_fg_patches == 4
    original = _trainer_positions(config)
    assert len(original) == 4

    label = zarr.open(str(dataset_dir / "labels" / "vol_ink.zarr"), mode="r+")
    label[:16, :16] = 0  # withdraw the labels of two of the four patches

    # The old list, two of whose patches now hold no label, must not be handed to the trainer.
    assert _trainer_positions(config) != original

    second = generate_patch_caches(config)
    assert (second.written_caches, second.total_fg_patches) == (1, 2)
    assert _trainer_positions(config) == {(0, 16, 0), (0, 16, 16)}
    assert len(list((dataset_dir / ".patches_cache").glob("patches_v*.json"))) == 1


def test_patch_cache_of_one_target_is_not_used_for_another(dataset_dir: Path) -> None:
    assert generate_patch_caches(_config(dataset_dir, "ink")).total_fg_patches == 4

    surface = generate_patch_caches(_config(dataset_dir, "surface"))

    assert (surface.written_caches, surface.total_fg_patches) == (1, 1)
    assert _trainer_positions(_config(dataset_dir, "surface")) == {(16, 16, 16)}


def test_patch_cache_without_a_fingerprint_is_still_accepted(dataset_dir: Path) -> None:
    config = _config(dataset_dir, "ink")
    generate_patch_caches(config)
    (cache_file,) = (dataset_dir / ".patches_cache").glob("patches_v*.json")
    payload = json.loads(cache_file.read_text())
    payload["metadata"]["cache_params"].pop("label_fingerprint", None)  # as written before this check
    cache_file.write_text(json.dumps(payload))

    second = generate_patch_caches(config)

    assert (second.written_caches, second.skipped_volumes) == (0, 1)
