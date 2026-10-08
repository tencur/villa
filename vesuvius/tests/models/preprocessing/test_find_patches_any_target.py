"""A volume labelled only for a non-first target must still contribute patches to the find_patches cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
import zarr

from vesuvius.models.preprocessing.patches.generate import generate_patch_caches


def _ome(path: Path, data: np.ndarray) -> None:
    g = zarr.open_group(str(path), mode="w")
    g.create_array("0", data=data, chunks=(32, 32, 32))
    g.create_array("1", data=data[::2, ::2, ::2], chunks=(16, 16, 16))


def test_volume_with_only_second_target_label_is_scanned(tmp_path: Path) -> None:
    img = np.full((64, 64, 64), 100, np.uint8)
    lab = np.ones((64, 64, 64), np.uint8)
    for vid in ("a", "b"):
        _ome(tmp_path / "images" / f"{vid}.zarr", img)
    _ome(tmp_path / "labels" / "a_ink.zarr", lab)
    _ome(tmp_path / "labels" / "b_damage.zarr", lab)
    target = {"out_channels": 1, "activation": "none", "losses": [{"name": "BCEWithLogitsLoss"}]}
    cfg = {
        "tr_setup": {"model_name": "t"},
        "tr_config": {"patch_size": [32, 32, 32]},
        "dataset_config": {"data_path": str(tmp_path), "allow_unlabeled_data": True, "min_labeled_ratio": 0.1,
                           "min_bbox_percent": 0.1, "valid_patch_find_resolution": 0,
                           "targets": {"ink": dict(target), "damage": dict(target)}},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = generate_patch_caches(path, force=True)
    assert result.total_fg_patches == 16, result   # 8 patches per volume, both volumes
