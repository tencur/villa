"""Checkpoints must record the training patch size for every architecture path.

vesuvius.predict rebuilds the network from model_config and uses model_config['patch_size'] (falling back to a
hard-coded 128^3) for its sliding window; the pretrained-backbone and Primus paths never stored it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from vesuvius.models.build.build_network_from_config import NetworkFromConfig
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.models.run.inference import Inferer


def _tiny_dinovol_model_config() -> dict:
    return {
        "model_type": "v2", "input_channels": 1, "global_crops_size": [16, 16, 16], "local_crops_size": [16, 16, 16],
        "patch_size": [8, 8, 8], "embed_dim": 48, "depth": 2, "num_heads": 4, "num_reg_tokens": 2, "mlp_ratio": 2.0,
        "drop_path_rate": 0.0, "qkv_fused": True,
    }


def _write_backbone_checkpoint(path: Path) -> None:
    backbone = build_dinovol_2_backbone(_tiny_dinovol_model_config())
    teacher_state = {f"backbone.{key}": value.cpu() for key, value in backbone.state_dict().items()}
    config = {"model": _tiny_dinovol_model_config(),
              "dataset": {"global_crop_size": [16, 16, 16], "local_crop_size": [16, 16, 16]}}
    torch.save({"teacher": teacher_state, "config": config}, path)


def test_pretrained_backbone_checkpoint_keeps_its_patch_size(tmp_path: Path) -> None:
    backbone_path = tmp_path / "tiny_dinovol.pt"
    _write_backbone_checkpoint(backbone_path)
    mgr = SimpleNamespace(
        train_patch_size=(16, 16, 16), train_batch_size=1, in_channels=1, autoconfigure=False,
        enable_deep_supervision=False, model_name="t", spacing=[1, 1, 1],
        targets={"surface": {"out_channels": 2, "activation": "none"}},
        model_config={"pretrained_backbone": str(backbone_path), "pretrained_decoder_type": "pixelshuffle_conv",
                      "input_shape": [16, 16, 16]},
    )
    model = NetworkFromConfig(mgr)
    assert tuple(model.final_config["patch_size"]) == (16, 16, 16)

    ckpt = tmp_path / "model.pth"
    torch.save({"model": model.state_dict(), "model_config": model.final_config}, ckpt)
    inferer = Inferer.__new__(Inferer)
    inferer.device = torch.device("cpu"); inferer.verbose = False
    inferer.model_normalization_scheme = None; inferer.model_intensity_properties = None
    info = inferer._load_train_py_model(ckpt)
    assert tuple(info["patch_size"]) == (16, 16, 16)
