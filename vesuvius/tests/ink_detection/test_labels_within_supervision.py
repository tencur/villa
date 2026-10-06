"""full_3d label projection must not supervise ink outside the supervision mask (e.g. held-out validation pixels)."""

from __future__ import annotations

import numpy as np

from vesuvius.ink_detection.data.geometry import project_labels_and_supervision


def _project(inklabels, supervision):
    positions = np.zeros((2, 2, 3), dtype=np.float32)
    positions[..., 0] = 2
    positions[..., 1] = np.array([[1, 1], [3, 3]])
    positions[..., 2] = np.array([[1, 3], [1, 3]])
    valid = np.ones((2, 2), dtype=bool)
    normals = np.zeros((2, 2, 3), dtype=np.float32)
    normals[..., 0] = 1
    return project_labels_and_supervision(
        positions_zyx=positions, valid_mask=valid,
        inklabels_flat=inklabels, supervision_flat=supervision,
        crop_bbox_zyx=(0, 0, 0, 5, 6, 6), normals_zyx=normals,
        label_half_thickness=1.0, background_half_thickness=1.0,
    )


def test_fully_held_out_support_projects_nothing():
    labels, supervision = _project(np.ones((2, 2), np.uint8), np.zeros((2, 2), np.uint8))
    assert labels.sum() == 0 and supervision.sum() == 0


def test_ink_outside_supervision_is_not_supervised():
    ink = np.ones((2, 2), np.uint8)
    sup = np.array([[1, 1], [0, 0]], np.uint8)          # bottom row held out
    labels, supervision = _project(ink, sup)
    full_labels, _ = _project(ink, np.ones((2, 2), np.uint8))
    assert 0 < labels.sum() < full_labels.sum()
    assert not np.any(labels[:, 3:, :])                   # nothing projected from the held-out row (y = 3)


def test_fully_supervised_unchanged():
    labels, supervision = _project(np.ones((2, 2), np.uint8), np.ones((2, 2), np.uint8))
    assert labels.sum() > 0 and np.array_equal(labels, supervision)
