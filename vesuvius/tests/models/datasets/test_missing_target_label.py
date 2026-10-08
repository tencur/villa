"""A target without a label file in a volume (allow_unlabeled_data) must not train as background when it has an ignore value."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.zarr_dataset import ZarrDataset

from tests.zarr_utils import create_v2_array


def _write(path: Path, data: np.ndarray) -> None:
    a = create_v2_array(path, shape=data.shape, chunks=(16, 16, 16), dtype=data.dtype)
    a[...] = data


def _sample(tmp_path, monkeypatch, damage_block: str):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    _write(tmp_path / "images" / "frag.zarr", rng.integers(1, 255, size=(16, 16, 16), dtype=np.uint8))
    ink = np.zeros((16, 16, 16), dtype=np.uint8)
    ink[:, :8] = 1
    _write(tmp_path / "labels" / "frag_ink.zarr", ink)   # no frag_damage.zarr
    cfg = tmp_path / "train.yaml"
    cfg.write_text("tr_config:\n  patch_size: [16, 16, 16]\n"
                   "dataset_config:\n  normalization_scheme: none\n  allow_unlabeled_data: true\n"
                   "  targets:\n    ink:\n      out_channels: 1\n      activation: none\n"
                   "    damage:\n      out_channels: 1\n      activation: none\n" + damage_block)
    mgr = ConfigManager(verbose=False)
    mgr.load_config(cfg)
    return ZarrDataset(mgr, is_training=False)[0]


def test_missing_label_is_ignored_when_target_has_ignore_value(tmp_path, monkeypatch):
    s = _sample(tmp_path, monkeypatch, "      ignore_label: 2\n")
    assert np.all(np.asarray(s["damage"]) == 2)
    assert np.asarray(s["ink"]).max() == 1          # the labelled target is unchanged
    assert not bool(s["is_unlabeled"])


def test_without_ignore_value_behaviour_is_unchanged(tmp_path, monkeypatch):
    s = _sample(tmp_path, monkeypatch, "")
    assert np.all(np.asarray(s["damage"]) == 0)
