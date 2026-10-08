"""tr_config.scheduler / scheduler_kwargs / gradient_clip / amp_dtype / no_amp and tr_setup.seed must reach the
attributes the trainer reads, and the CLI must override them only when a flag is given explicitly."""

from __future__ import annotations

from pathlib import Path

import yaml

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.training.cli import build_parser


def _load(tmp_path: Path, tr_setup: dict, tr_config: dict) -> ConfigManager:
    cfg = {
        "tr_setup": {"model_name": "t", **tr_setup},
        "tr_config": {"patch_size": [16, 16, 16], **tr_config},
        "dataset_config": {"targets": {"ink": {"out_channels": 1, "activation": "none",
                                                "losses": [{"name": "BCEWithLogitsLoss"}]}}},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    mgr = ConfigManager(False)
    mgr.load_config(str(path))
    return mgr


def test_yaml_training_keys_are_promoted(tmp_path: Path) -> None:
    mgr = _load(
        tmp_path,
        {"seed": 7},
        {"scheduler": "cosine_warmup", "scheduler_kwargs": {"warmup_steps": 5},
         "gradient_clip": 1.0, "amp_dtype": "bfloat16", "no_amp": True},
    )
    assert mgr.scheduler == "cosine_warmup"
    assert mgr.scheduler_kwargs == {"warmup_steps": 5}
    assert mgr.gradient_clip == 1.0
    assert mgr.amp_dtype == "bfloat16"
    assert mgr.no_amp is True
    assert mgr.seed == 7


def test_yaml_training_keys_keep_previous_defaults(tmp_path: Path) -> None:
    mgr = _load(tmp_path, {}, {})
    assert (mgr.scheduler, mgr.scheduler_kwargs, mgr.gradient_clip, mgr.amp_dtype, mgr.no_amp, mgr.seed) == (
        "poly", {}, 12.0, "float16", False, 42)


def test_cli_flags_default_to_unset() -> None:
    args = build_parser().parse_args(["--config", "cfg.yaml"])
    assert args.seed is None and args.grad_clip is None and args.amp_dtype is None
