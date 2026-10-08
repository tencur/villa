"""
Patch cache generation for OME-Zarr datasets.

This module provides standalone patch cache generation using direct zarr reading
and the find_valid_patches() utility. It follows the same pattern as ZarrDataset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import zarr

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.find_valid_patches import find_valid_patches
from .cache import (
    PatchCacheData,
    PatchCacheParams,
    PatchEntry,
    build_cache_params,
    load_patch_cache,
    save_patch_cache,
)

logger = logging.getLogger(__name__)


@dataclass
class VolumeInfo:
    """Metadata for a discovered OME-Zarr volume."""

    volume_id: str
    image_path: Path
    label_paths: Dict[str, Path]


@dataclass
class PatchCacheResult:
    """Result summary from patch cache generation."""

    total_volumes: int
    scanned_volumes: int
    skipped_volumes: int
    written_caches: int
    total_fg_patches: int
    total_bg_patches: int
    total_unlabeled_fg_patches: int


def generate_patch_caches(
    config_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> PatchCacheResult:
    """
    Generate patch caches for all volumes in a dataset.

    Parameters
    ----------
    config_path : Path
        Path to training config YAML file.
    cache_dir : Path, optional
        Override cache directory. Defaults to data_path/.patches_cache
    force : bool
        Force regeneration even if cache exists.

    Returns
    -------
    PatchCacheResult
        Summary of cache generation results.
    """
    # Load config
    mgr = ConfigManager(verbose=True)
    mgr.load_config(config_path)

    data_path = Path(mgr.data_path)
    patch_size = tuple(int(v) for v in mgr.train_patch_size)

    # Resolve target names
    target_names = _resolve_target_names(mgr)
    logger.info("Target names: %s", target_names)

    # Resolve cache directory
    if cache_dir is None:
        cache_dir = data_path / ".patches_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cache directory: %s", cache_dir)

    # Discover volumes
    volumes = _discover_volumes(data_path, target_names)
    if not volumes:
        logger.warning("No volumes found in %s", data_path)
        return PatchCacheResult(0, 0, 0, 0, 0, 0, 0)

    logger.info("Discovered %d volumes", len(volumes))

    # Build cache params
    volume_ids = [v.volume_id for v in volumes]
    first_target = target_names[0]
    # Each volume is scanned with the first target (in config order) it is labelled for, so a
    # volume annotated only for a later task is still scanned instead of being dropped.
    scan_targets = [scan_label_target(v.label_paths, target_names) for v in volumes]
    ignore_label = resolve_ignore_label(target_names, mgr)
    # Per volume, the ignore label of the target whose label is scanned; the cache key carries
    # the first target's, as every volume with a first-target label is scanned with it.
    volume_ignore_labels = [
        resolve_ignore_label([target], mgr) if target is not None else None for target in scan_targets
    ]
    # Collected whenever the ignore label is known, so the cache does not depend on bg_sampling_enabled;
    # the dataset loads them only when BG sampling is on.
    collect_bg_only = any(value is not None for value in volume_ignore_labels)
    cache_params = build_cache_params(
        data_path=data_path,
        volume_ids=volume_ids,
        patch_size=patch_size,
        min_labeled_ratio=float(getattr(mgr, "min_labeled_ratio", 0.10)),
        bbox_threshold=float(getattr(mgr, "min_bbox_percent", 0.95)),
        valid_patch_find_resolution=int(getattr(mgr, "valid_patch_find_resolution", 1)),
        valid_patch_value=_resolve_valid_patch_value(target_names, mgr),
        unlabeled_fg_enabled=bool(getattr(mgr, "unlabeled_foreground_enabled", False)),
        unlabeled_fg_threshold=float(getattr(mgr, "unlabeled_foreground_threshold", 0.05)),
        unlabeled_fg_bbox_threshold=float(
            getattr(mgr, "unlabeled_foreground_bbox_threshold", 0.15)
        ),
        label_paths=[
            v.label_paths.get(target) if target is not None else None
            for v, target in zip(volumes, scan_targets)
        ],
        ignore_label=ignore_label,
    )

    # Check if cache already exists
    if not force:
        cached = load_patch_cache(cache_dir, cache_params)
        if cached is not None:
            logger.info("Cache already exists with %d FG patches, skipping", len(cached.fg_patches))
            return PatchCacheResult(
                total_volumes=len(volumes),
                scanned_volumes=0,
                skipped_volumes=len(volumes),
                written_caches=0,
                total_fg_patches=len(cached.fg_patches),
                total_bg_patches=len(cached.bg_patches),
                total_unlabeled_fg_patches=len(cached.unlabeled_fg_patches),
            )

    # Prepare arrays for find_valid_patches
    label_arrays: List[Optional[zarr.Array]] = []
    label_names: List[str] = []
    image_arrays: List[Optional[zarr.Array]] = []

    unlabeled_fg_enabled = bool(getattr(mgr, "unlabeled_foreground_enabled", False))
    unlabeled_fg_volume_ids = set(getattr(mgr, "unlabeled_foreground_volumes", []) or [])

    for vol, scan_target in zip(volumes, scan_targets):
        # Label array - pass zarr Group for multi-resolution support. Use the first target
        # (in config order) that this volume is labelled for, so a volume annotated only for
        # a later task is still scanned instead of being dropped from the cache.
        label_path = vol.label_paths.get(scan_target) if scan_target is not None else None
        if scan_target is not None and scan_target != first_target:
            logger.info(
                "Volume '%s' has no '%s' label; finding patches from its '%s' label",
                vol.volume_id, first_target, scan_target,
            )
        if label_path and label_path.exists():
            try:
                label_arrays.append(zarr.open(label_path, mode="r"))
            except Exception as e:
                logger.warning("Failed to open label %s: %s", label_path, e)
                label_arrays.append(None)
        else:
            label_arrays.append(None)

        label_names.append(vol.volume_id)

        # Image array for unlabeled FG detection
        if unlabeled_fg_enabled:
            should_use = not unlabeled_fg_volume_ids or vol.volume_id in unlabeled_fg_volume_ids
            if should_use:
                try:
                    image_arrays.append(zarr.open(vol.image_path, mode="r"))
                except Exception as e:
                    logger.warning("Failed to open image %s: %s", vol.image_path, e)
                    image_arrays.append(None)
            else:
                image_arrays.append(None)
        else:
            image_arrays.append(None)

    # Run find_valid_patches
    logger.info("Running patch validation...")
    result = find_valid_patches(
        label_arrays=label_arrays,
        label_names=label_names,
        patch_size=patch_size,
        bbox_threshold=cache_params.bbox_threshold,
        label_threshold=cache_params.min_labeled_ratio,
        valid_patch_find_resolution=cache_params.valid_patch_find_resolution,
        valid_patch_values=(
            [cache_params.valid_patch_value] * len(label_arrays)
            if cache_params.valid_patch_value is not None
            else None
        ),
        ignore_labels=volume_ignore_labels if collect_bg_only else None,
        collect_bg_only=collect_bg_only,
        image_arrays=image_arrays if unlabeled_fg_enabled else None,
        collect_unlabeled_fg=unlabeled_fg_enabled,
        unlabeled_fg_threshold=cache_params.unlabeled_fg_threshold,
        unlabeled_fg_bbox_threshold=cache_params.unlabeled_fg_bbox_threshold,
    )

    # Convert results to cache format - preserve volume info
    fg_patches = [
        PatchEntry(
            volume_idx=p["volume_idx"],
            volume_name=p["volume_name"],
            position=tuple(p["start_pos"]),
        )
        for p in result["fg_patches"]
    ]
    bg_patches = [
        PatchEntry(
            volume_idx=p["volume_idx"],
            volume_name=p["volume_name"],
            position=tuple(p["start_pos"]),
        )
        for p in result["bg_patches"]
    ]
    unlabeled_fg_patches = [
        PatchEntry(
            volume_idx=p["volume_idx"],
            volume_name=p["volume_name"],
            position=tuple(p["start_pos"]),
        )
        for p in result["unlabeled_fg_patches"]
    ]

    cache_data = PatchCacheData(
        fg_patches=fg_patches,
        bg_patches=bg_patches,
        unlabeled_fg_patches=unlabeled_fg_patches,
    )

    # Save cache
    cache_file = save_patch_cache(cache_dir, cache_params, cache_data)
    logger.info("Saved cache to %s", cache_file)

    return PatchCacheResult(
        total_volumes=len(volumes),
        scanned_volumes=len(volumes),
        skipped_volumes=0,
        written_caches=1,
        total_fg_patches=len(fg_patches),
        total_bg_patches=len(bg_patches),
        total_unlabeled_fg_patches=len(unlabeled_fg_patches),
    )


def _resolve_target_names(mgr) -> List[str]:
    """Extract primary (non-auxiliary) target names from config."""
    targets = getattr(mgr, "targets", {})
    if not targets:
        return ["ink"]  # Default target

    names = [
        name
        for name, info in targets.items()
        if not (info or {}).get("auxiliary_task", False)
    ]
    return names if names else ["ink"]


def _resolve_valid_patch_value(
    target_names: List[str],
    mgr,
) -> Optional[Union[int, float]]:
    """Extract valid_patch_value from target config, with dataset-level fallback."""
    targets = getattr(mgr, "targets", {})
    for target in target_names:
        info = targets.get(target) or {}
        value = info.get("valid_patch_value")
        if value is not None:
            return value
    dataset_cfg = getattr(mgr, "dataset_config", {}) or {}
    dataset_value = dataset_cfg.get("valid_patch_value")
    if dataset_value is not None:
        return dataset_value
    return None


def scan_label_target(
    label_paths: Dict[str, Optional[Path]],
    target_names: List[str],
) -> Optional[str]:
    """Name of the target whose label the patch finder scans for a volume: the first target,
    in config order, that the volume has a label for (``None`` when it has none). Used by
    vesuvius.find_patches and by the training dataset's cache lookup, so both fingerprint the
    same label assets.
    """
    for target in target_names:
        path = label_paths.get(target)
        if path is not None and Path(path).exists():
            return target
    return None


def resolve_ignore_label(
    target_names: List[str],
    mgr,
) -> Optional[Union[int, float]]:
    """Label value that marks unannotated voxels in the first target's labels (the ones patch
    finding scans), resolved in the same key order the trainer's loss masking uses.
    """
    if not target_names:
        return None
    info = (getattr(mgr, "targets", {}) or {}).get(target_names[0]) or {}
    for key in ("ignore_index", "ignore_label", "ignore_value"):
        value = info.get(key)
        if value is not None:
            return value
    return None


def _discover_volumes(
    data_path: Path,
    target_names: List[str],
) -> List[VolumeInfo]:
    """
    Discover OME-Zarr volumes following ZarrDataset pattern.

    Expected directory structure:
        data_path/
            images/
                volume1.zarr/
                volume2.zarr/
            labels/
                volume1_target.zarr/
                volume2_target.zarr/
    """
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"

    if not images_dir.exists():
        logger.warning("Images directory not found: %s", images_dir)
        return []

    # Find all image zarrs
    image_zarrs = {
        p.stem: p
        for p in images_dir.iterdir()
        if p.is_dir() and p.suffix == ".zarr"
    }

    if not image_zarrs:
        logger.warning("No .zarr directories found in %s", images_dir)
        return []

    volumes = []
    for volume_id, image_path in sorted(image_zarrs.items()):
        label_paths: Dict[str, Path] = {}

        for target in target_names:
            label_path = labels_dir / f"{volume_id}_{target}.zarr"
            if label_path.exists():
                label_paths[target] = label_path

        volumes.append(
            VolumeInfo(
                volume_id=volume_id,
                image_path=image_path,
                label_paths=label_paths,
            )
        )

    return volumes
