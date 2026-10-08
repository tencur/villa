"""mean_teacher_config.ema_decay must reach the mean-teacher trainers.

ConfigManager flattens mean_teacher_config onto mgr and then sets mgr.ema_decay for the generic model EMA
(ema_config.decay, default 0.999), which overwrote the teacher's decay: the trainers always ran at 0.999
whatever the config said, and their own 0.99 default never applied.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.training.trainers.semi_supervised.ramps import mean_teacher_ema_decay


def _mgr(tmp_path: Path, monkeypatch, extra: str) -> ConfigManager:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("tr_config:\n  patch_size: [16, 16, 16]\n"
                   "dataset_config:\n  targets:\n    ink:\n      activation: none\n" + extra)
    mgr = ConfigManager(verbose=False)
    mgr.load_config(cfg)
    return mgr


def test_configured_teacher_decay_is_used(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch, "mean_teacher_config:\n  ema_decay: 0.9\n")
    assert mean_teacher_ema_decay(mgr) == pytest.approx(0.9)
    assert mgr.ema_decay == pytest.approx(0.999)  # the generic model EMA keeps its own value


def test_trainer_default_applies_when_not_configured(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch, "mean_teacher_config:\n  consistency_rampup: 200.0\n")
    assert mean_teacher_ema_decay(mgr) == pytest.approx(0.99)


def test_both_trainers_use_it():
    import inspect
    from vesuvius.models.training.trainers.semi_supervised import (
        train_mean_teacher, train_uncertainty_aware_mean_teacher)
    for module in (train_mean_teacher, train_uncertainty_aware_mean_teacher):
        assert "self.ema_decay = mean_teacher_ema_decay(mgr)" in inspect.getsource(module)
