"""A binary target stored as a grayscale mask (0/255) must not reach BCE as a target of 255: the dataset raises."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.zarr_dataset import ZarrDataset

from tests.zarr_utils import create_v2_array


def _write(path: Path, data: np.ndarray) -> None:
    a = create_v2_array(path, shape=data.shape, chunks=(16, 16, 16), dtype=data.dtype)
    a[...] = data


def _sample(tmp_path, monkeypatch, target_block: str, label: np.ndarray):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    _write(tmp_path / "images" / "vol.zarr", rng.integers(1, 255, size=(16, 16, 16), dtype=np.uint8))
    _write(tmp_path / "labels" / "vol_ink.zarr", label)
    cfg = tmp_path / "train.yaml"
    cfg.write_text("tr_config:\n  patch_size: [16, 16, 16]\n"
                   "dataset_config:\n  normalization_scheme: none\n  targets:\n    ink:\n" + target_block)
    mgr = ConfigManager(verbose=False)
    mgr.load_config(cfg)
    return np.asarray(ZarrDataset(mgr, is_training=False)[0]["ink"])


def _label(values=(0, 255)):
    lab = np.full((16, 16, 16), values[0], dtype=np.uint8)
    lab[:, :8] = values[1]
    return lab


BCE = "      out_channels: 1\n      activation: sigmoid\n      losses:\n        - name: BCEWithLogitsLoss\n          weight: 1.0\n"


def test_grayscale_mask_is_rejected_for_bce(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        _sample(tmp_path, monkeypatch, BCE, _label())


def test_zero_one_label_passes(tmp_path, monkeypatch):
    got = _sample(tmp_path, monkeypatch, BCE, _label((0, 1)))
    assert set(np.unique(got).tolist()) == {0.0, 1.0}


def test_ignore_value_is_kept(tmp_path, monkeypatch):
    lab = _label((0, 1))
    lab[:, 12:] = 2
    got = _sample(tmp_path, monkeypatch, BCE + "      ignore_label: 2\n", lab)
    assert set(np.unique(got).tolist()) == {0.0, 1.0, 2.0}


def test_regression_target_is_not_binarized(tmp_path, monkeypatch):
    block = "      out_channels: 1\n      activation: none\n      losses:\n        - name: MSELoss\n          weight: 1.0\n"
    got = _sample(tmp_path, monkeypatch, block, _label((0, 37)))
    assert got.max() == 37.0


def test_soft_targets_in_unit_interval_pass(tmp_path, monkeypatch):
    lab = np.zeros((16, 16, 16), dtype=np.float32)
    lab[:, :8] = 0.7
    got = _sample(tmp_path, monkeypatch, BCE, lab)
    assert np.isclose(got.max(), 0.7)



def test_grayscale_mask_is_rejected_when_bce_is_mixed_with_another_loss(tmp_path, monkeypatch):
    block = BCE + "        - name: MSELoss\n          weight: 1.0\n"
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        _sample(tmp_path, monkeypatch, block, _label())


def test_ignore_index_on_the_loss_entry_is_kept(tmp_path, monkeypatch):
    # The trainer passes extra keys of a loss entry to that loss, so this masks 2 just like a target-level ignore.
    lab = _label((0, 1))
    lab[:, 12:] = 2
    block = BCE + "          ignore_index: 2\n"
    got = _sample(tmp_path, monkeypatch, block, lab)
    assert set(np.unique(got).tolist()) == {0.0, 1.0, 2.0}
