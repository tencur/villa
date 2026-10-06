"""dataset_config.ome_zarr_resolution must select the pyramid level the training dataset reads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.preprocessing.patches import generate_patch_caches


def _pyramid(path: Path, level0: np.ndarray) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.create_array("0", data=level0, chunks=(16, 16, 16))
    root.create_array("1", data=level0[::2, ::2, ::2].copy(), chunks=(16, 16, 16))


def _dataset(tmp_path: Path, monkeypatch, level: int) -> ZarrDataset:
    monkeypatch.chdir(tmp_path)
    z = np.arange(64, dtype=np.uint8)[:, None, None]
    image = np.broadcast_to(z * 3, (64, 64, 64)).astype(np.uint8)
    label = np.zeros((64, 64, 64), dtype=np.uint8)
    label[:, 20:44, 20:44] = 1
    _pyramid(tmp_path / "images" / "vol.zarr", image)
    _pyramid(tmp_path / "labels" / "vol_ink.zarr", label)
    cfg = tmp_path / "train.yaml"
    cfg.write_text("tr_config:\n  patch_size: [16, 16, 16]\n"
                   f"dataset_config:\n  ome_zarr_resolution: {level}\n  normalization_scheme: none\n"
                   "  min_labeled_ratio: 0.05\n  min_bbox_percent: 0.05\n  valid_patch_find_resolution: 0\n"
                   "  targets:\n    ink:\n      activation: none\n")
    generate_patch_caches(cfg)
    mgr = ConfigManager(verbose=False)
    mgr.load_config(cfg)
    return ZarrDataset(mgr, is_training=False)


def test_level_one_is_read_and_cached_positions_are_mapped(tmp_path, monkeypatch):
    d = _dataset(tmp_path, monkeypatch, 1)
    assert tuple(d._volumes[0].spatial_shape) == (32, 32, 32)
    level1 = zarr.open(str(tmp_path / "images" / "vol.zarr"), mode="r")["1"][:]
    label1 = zarr.open(str(tmp_path / "labels" / "vol_ink.zarr"), mode="r")["1"][:]
    assert len(d) > 0
    for i in range(len(d)):
        z, y, x = d.valid_patches[i].position
        assert max(z, y, x) + 16 <= 32
        item = d[i]
        np.testing.assert_array_equal(np.asarray(item["image"])[0], level1[z:z + 16, y:y + 16, x:x + 16])
        np.testing.assert_array_equal(np.asarray(item["ink"])[0], label1[z:z + 16, y:y + 16, x:x + 16])


def test_level_zero_unchanged(tmp_path, monkeypatch):
    d = _dataset(tmp_path, monkeypatch, 0)
    assert tuple(d._volumes[0].spatial_shape) == (64, 64, 64)
