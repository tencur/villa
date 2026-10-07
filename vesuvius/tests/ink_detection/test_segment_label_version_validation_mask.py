"""An explicit label_version must not drop the segment's validation mask when that mask is unversioned."""

from __future__ import annotations

from pathlib import Path

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.data.segment import discover_segment_labels
from vesuvius.ink_detection.types import Segment


def _segment(tmp_path: Path, label_version: str | None) -> Segment:
    seg_dir = tmp_path / "seg"
    for name in ("seg_inklabels_v2.zarr", "seg_supervision_mask_v2.zarr", "seg_inklabels.zarr",
                 "seg_supervision_mask.zarr", "seg_validation_mask.zarr"):
        (seg_dir / name).mkdir(parents=True, exist_ok=True)
    mapping = {"patch_size": [8, 8, 8], "patch_overlap": 0.5, "patch_min_labeled_coverage": 0.0,
               "datasets": [{"segments_path": str(tmp_path), "volume_scale": 0}]}
    if label_version:
        mapping["label_version"] = label_version
    cfg = InkDataConfig.from_mapping(mapping)
    return Segment(cfg, cfg.datasets[0], 0, "seg", seg_dir, "seg", "vol")


def test_default_resolution_keeps_the_validation_mask(tmp_path: Path) -> None:
    seg = discover_segment_labels(_segment(tmp_path, None))
    assert seg.inklabels.name == "seg_inklabels_v2.zarr"
    assert seg.validation_mask is not None and seg.validation_mask.name == "seg_validation_mask.zarr"


def test_explicit_version_keeps_the_unversioned_validation_mask(tmp_path: Path) -> None:
    seg = discover_segment_labels(_segment(tmp_path, "v2"))
    assert seg.inklabels.name == "seg_inklabels_v2.zarr"
    assert seg.supervision_mask.name == "seg_supervision_mask_v2.zarr"
    assert seg.validation_mask is not None, "the held-out region must survive version pinning"
    assert seg.validation_mask.name == "seg_validation_mask.zarr"
