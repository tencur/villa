"""Multiclass finalize must encode a class the same way in every chunk.

The old code min-max rescaled each chunk by its own content: with three classes, class 1 became 127 in a chunk
that also held class 2, 255 in a chunk without class 2, and a chunk uniformly of one non-zero class was dropped as
empty (read back as class 0). The fixed encoding is class_id * 255 / (num_classes - 1), which keeps the two-class
output at 0/255.
"""

from __future__ import annotations

import numpy as np

from vesuvius.models.run.finalize_outputs import FinalizeConfig, apply_finalization


def _logits_for(classes: np.ndarray, n_classes: int) -> np.ndarray:
    logits = np.full((n_classes,) + classes.shape, -5.0, dtype=np.float32)
    for c in range(n_classes):
        logits[c][classes == c] = 5.0
    return logits


def test_three_class_argmax_encoding_does_not_depend_on_chunk_content():
    shape = (4, 4, 4)
    with_all = np.zeros(shape, np.int64); with_all[0] = 1; with_all[1] = 2
    without_two = np.zeros(shape, np.int64); without_two[0] = 1
    uniform_one = np.ones(shape, np.int64)
    cfg = FinalizeConfig("multiclass", 0.5)
    out_all, _ = apply_finalization(_logits_for(with_all, 3), 3, cfg)
    out_without, _ = apply_finalization(_logits_for(without_two, 3), 3, cfg)
    out_uniform, empty = apply_finalization(_logits_for(uniform_one, 3), 3, cfg)
    assert out_all[0, 0, 0, 0] == out_without[0, 0, 0, 0] == 128, "class 1 must have one encoding"
    assert out_all[0, 1, 0, 0] == 255
    assert not empty and np.all(out_uniform == 128), "a uniform class chunk is real output"


def test_two_class_output_stays_0_255_and_uniform_foreground_is_written():
    cfg = FinalizeConfig("multiclass", 0.5)
    mixed = np.zeros((4, 4, 4), np.int64); mixed[0] = 1
    out_mixed, _ = apply_finalization(_logits_for(mixed, 2), 2, cfg)
    assert set(np.unique(out_mixed).tolist()) == {0, 255}
    out_fg, empty = apply_finalization(_logits_for(np.ones((4, 4, 4), np.int64), 2), 2, cfg)
    assert not empty and np.all(out_fg == 255)


def test_softmax_channels_are_probabilities_times_255():
    classes = np.zeros((4, 4, 4), np.int64); classes[0] = 2
    out, _ = apply_finalization(_logits_for(classes, 3), 3, FinalizeConfig("multiclass", None))
    assert out.shape[0] == 4
    assert out[2, 0, 0, 0] == 255 and out[0, 0, 0, 0] == 0
    assert out[3, 0, 0, 0] == 255 and out[3, 1, 0, 0] == 0
