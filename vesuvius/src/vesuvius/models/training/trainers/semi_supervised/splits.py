"""Train/validation split helpers shared by the semi-supervised trainers."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple


def validation_shares_training_source(train_dataset, val_dataset) -> bool:
    """True when validation patches come from the same data as training (no separate --val-dir)."""
    if val_dataset is None or val_dataset is train_dataset:
        return True
    train_path = getattr(train_dataset, "data_path", None)
    val_path = getattr(val_dataset, "data_path", None)
    return bool(train_path and val_path and train_path == val_path)


def hold_out_labeled_validation(
    labeled_idx: Sequence[int], tr_val_split: float, min_train: int = 1
) -> Tuple[List[int], List[int]]:
    """Split shuffled labeled patch indices into (train, validation) without overlap.

    Validation takes the last floor((1 - tr_val_split) * n) labeled patches (at least one when there are two or
    more), leaving at least ``min_train`` for training. Unlabeled patches are never used for validation: they have
    no labels to score against.
    """
    labeled = list(labeled_idx)
    n = len(labeled)
    if n < 2:
        return labeled, []
    n_val = int(math.floor((1.0 - float(tr_val_split)) * n + 1e-9))
    n_val = max(1, min(n_val, n - max(1, int(min_train))))
    if n_val <= 0:
        return labeled, []
    return labeled[:-n_val], labeled[-n_val:]
