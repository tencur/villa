"""`vesuvius.predict --overlap` is the fraction of a patch shared with its neighbour."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vesuvius.models.run.inference as inference
from vesuvius.models.run.inference import Inferer
from vesuvius.utils.models.helpers import compute_steps_for_sliding_window


class _DatasetReached(Exception):
    pass


def _step_given_to_dataset(monkeypatch, overlap: float) -> float:
    """The step_size Inferer passes to its dataset for a requested overlap."""
    seen = {}

    def recording_dataset(**kwargs):
        seen.update(kwargs)
        raise _DatasetReached

    monkeypatch.setattr(inference, "VCDataset", recording_dataset)
    inferer = Inferer.__new__(Inferer)
    for name in (
        "model_normalization_scheme", "model_intensity_properties", "input_format",
        "scroll_id", "segment_id", "energy", "resolution", "bbox",
    ):
        setattr(inferer, name, None)
    inferer.normalization_scheme = "none"
    inferer.verbose = False
    inferer.input = "volume.zarr"
    inferer.patch_size = (128, 128, 128)
    inferer.num_parts = 1
    inferer.part_id = 0
    inferer.skip_empty_patches = True
    inferer.input_anon = False
    inferer.read_retries = 0
    inferer.device = torch.device("cpu")
    inferer.overlap = overlap
    with pytest.raises(_DatasetReached):
        inferer._create_dataset_and_loader()
    return seen["step_size"]


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
def test_requested_overlap_is_the_overlap_between_neighbouring_patches(monkeypatch, overlap):
    patch = 128
    positions = compute_steps_for_sliding_window(
        4096, patch, _step_given_to_dataset(monkeypatch, overlap)
    )
    shared = 1.0 - np.diff(positions) / patch
    # Positions are spread evenly over the volume, so the overlap can only round up.
    assert shared.min() >= overlap - 1e-9
    assert shared.max() <= overlap + 0.02
