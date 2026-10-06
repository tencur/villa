"""Validation metrics must score every class a target declares."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from vesuvius.models.evaluation.iou_dice import IOUDiceMetric
from vesuvius.models.training.train import BaseTrainer


def _metrics(target):
    trainer = SimpleNamespace(mgr=SimpleNamespace(targets={"t": target}))
    return BaseTrainer._initialize_evaluation_metrics(trainer)["t"]


def _dice(metrics):
    return next(m for m in metrics if isinstance(m, IOUDiceMetric))


def test_three_class_target_is_scored_on_all_three_classes():
    dice = _dice(_metrics({"out_channels": 3, "ignore_label": 3}))
    assert dice.num_classes == 3

    gt = np.zeros((4, 4, 4), np.int64); gt[0] = 1; gt[1] = 2
    pred_classes = np.where(gt == 2, 0, gt)  # every class-2 voxel predicted as background
    pred = np.stack([(pred_classes == c) * 10.0 - 5.0 for c in range(3)])[None].astype(np.float32)
    scores = dice.compute(torch.from_numpy(pred), torch.from_numpy(gt[None, None]))
    assert scores["dice_class_2"] < 0.01
    assert scores["mean_dice"] < 0.7


def test_single_channel_and_explicit_counts():
    assert _dice(_metrics({"out_channels": 1})).num_classes == 2
    assert _dice(_metrics({"out_channels": 2})).num_classes == 2
    assert _dice(_metrics({"out_channels": 3, "num_classes": 4})).num_classes == 4
