"""Semi-supervised trainers must validate on held-out labeled patches when validation shares the training data."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vesuvius.models.training.trainers.semi_supervised.train_mean_teacher import TrainMeanTeacher
from vesuvius.models.training.trainers.semi_supervised.train_uncertainty_aware_mean_teacher import (
    TrainUncertaintyAwareMeanTeacher,
)


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, n_labeled=40, n_unlabeled=40, data_path="/data/scroll"):
        self.n_labeled, self.n_unlabeled = n_labeled, n_unlabeled
        self.data_path = data_path

    def __len__(self):
        return self.n_labeled + self.n_unlabeled

    def __getitem__(self, i):
        return {"image": torch.zeros(1, 4, 4, 4)}

    def get_labeled_unlabeled_patch_indices(self):
        return list(range(self.n_labeled)), list(range(self.n_labeled, len(self)))


def _trainer(cls):
    t = cls.__new__(cls)
    t.mgr = SimpleNamespace(seed=0, tr_val_split=0.8, train_batch_size=4, verbose=False,
                            train_num_dataloader_workers=0)
    t.labeled_batch_size = 2
    t.num_labeled = None
    t.labeled_ratio = 1.0
    t.device = torch.device("cpu")
    return t


@pytest.mark.parametrize("cls", [TrainMeanTeacher, TrainUncertaintyAwareMeanTeacher])
def test_same_source_validation_is_held_out_and_labeled(cls):
    train_ds, val_ds = FakeDataset(), FakeDataset()          # separate objects, same data_path (no --val-dir)
    t = _trainer(cls)
    _, _, train_idx, val_idx = t._configure_dataloaders(train_ds, val_ds)
    assert val_idx, "validation must not be empty"
    assert not set(val_idx) & set(t.labeled_indices)          # never trained on
    assert set(val_idx) <= set(range(train_ds.n_labeled))     # never an unlabeled patch
    assert len(val_idx) == 8                                  # (1 - 0.8) * 40


@pytest.mark.parametrize("cls", [TrainMeanTeacher, TrainUncertaintyAwareMeanTeacher])
def test_external_validation_uses_the_whole_validation_set(cls):
    train_ds, val_ds = FakeDataset(), FakeDataset(n_labeled=10, n_unlabeled=0, data_path="/data/other")
    t = _trainer(cls)
    _, _, _, val_idx = t._configure_dataloaders(train_ds, val_ds)
    assert list(val_idx) == list(range(10))


def test_hold_out_keeps_enough_for_training():
    from vesuvius.models.training.trainers.semi_supervised.splits import hold_out_labeled_validation
    train, val = hold_out_labeled_validation(list(range(3)), 0.1, min_train=2)
    assert len(train) == 2 and val == [2]
