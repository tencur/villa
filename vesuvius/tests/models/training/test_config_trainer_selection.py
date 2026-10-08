"""vesuvius.train must honour tr_config.trainer when --trainer is not given."""

from __future__ import annotations

from pathlib import Path

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.training.cli import build_parser, resolve_trainer_name

import vesuvius.models.configuration as _configuration

SHIPPED = Path(_configuration.__file__).parent / "semi_supervised" / "uncertainty_aware_mean_teacher.yaml"


def _mgr(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    mgr = ConfigManager(verbose=False)
    mgr.load_config(path)
    return mgr


def test_config_trainer_is_used_without_flag(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch, SHIPPED)
    args = build_parser().parse_args(["--config", str(SHIPPED)])
    assert resolve_trainer_name(args.trainer, mgr) == "uncertainty_aware_mean_teacher"


def test_flag_still_wins(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, monkeypatch, SHIPPED)
    args = build_parser().parse_args(["--config", str(SHIPPED), "--trainer", "base"])
    assert resolve_trainer_name(args.trainer, mgr) == "base"


def test_default_is_base(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("tr_config:\n  patch_size: [16, 16, 16]\ndataset_config:\n  targets:\n    ink:\n      activation: none\n")
    mgr = _mgr(tmp_path, monkeypatch, cfg)
    args = build_parser().parse_args(["--config", str(cfg)])
    assert resolve_trainer_name(args.trainer, mgr) == "base"
