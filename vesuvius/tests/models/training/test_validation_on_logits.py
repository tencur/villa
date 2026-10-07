"""The trainer validates in eval mode; the model it builds must still return logits there.

NetworkFromConfig applies each target's configured activation whenever the module is not training, so without
this the validation loss (BCEWithLogitsLoss, CrossEntropyLoss, Dice) and the metrics would see probabilities.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vesuvius.models.training.train import BaseTrainer


def _mgr(activation: str, out_channels: int) -> SimpleNamespace:
    model_config = {
        "basic_encoder_block": "BasicBlockD",
        "basic_decoder_block": "ConvBlock",
        "bottleneck_block": "BasicBlockD",
        "features_per_stage": [4, 8],
        "n_stages": 2,
        "n_blocks_per_stage": [1, 1],
        "n_conv_per_stage_decoder": [1],
        "kernel_sizes": [[3, 3, 3], [3, 3, 3]],
        "strides": [[1, 1, 1], [2, 2, 2]],
        "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2]],
        "separate_decoders": False,
        "autoconfigure": False,
    }
    return SimpleNamespace(
        model_config=model_config,
        targets={"ink": {"out_channels": out_channels, "activation": activation}},
        train_patch_size=(16, 16, 16),
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        enable_deep_supervision=False,
        model_name="Model",
        spacing=[1, 1, 1],
    )


def _out(model, x):
    with torch.inference_mode():
        o = model(x)
    return o["ink"] if isinstance(o, dict) else o


@pytest.mark.parametrize("activation,out_channels", [("sigmoid", 1), ("softmax", 2)])
def test_trainer_model_returns_logits_in_eval_mode(activation, out_channels):
    torch.manual_seed(0)
    trainer = BaseTrainer(mgr=_mgr(activation, out_channels), verbose=False)
    model = trainer._build_model()
    x = torch.randn(1, 1, 16, 16, 16) * 3
    model.train()
    train_out = _out(model, x)
    model.eval()
    eval_out = _out(model, x)
    assert torch.allclose(train_out, eval_out, atol=1e-5), "eval-mode output must be the same logits as train mode"
    assert eval_out.min() < 0


@pytest.mark.parametrize("weights_only", [False, True], ids=["full_resume", "weights_only"])
def test_resumed_model_still_returns_logits(tmp_path, weights_only):
    """A full resume rebuilds the network from the checkpoint config; that rebuild must return logits too."""
    from vesuvius.models.training.lr_schedulers import get_scheduler
    from vesuvius.models.utilities.load_checkpoint import load_checkpoint

    def setup():
        mgr = _mgr("sigmoid", 1)
        mgr.optimizer, mgr.initial_lr, mgr.weight_decay = "AdamW", 1e-3, 0.0
        mgr.scheduler, mgr.max_epoch, mgr.scheduler_kwargs = "poly", 10, {}
        model = BaseTrainer(mgr=mgr, verbose=False)._build_model()
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = get_scheduler(scheduler_type="poly", optimizer=optimizer, initial_lr=1e-3, max_steps=10)
        return mgr, model, optimizer, scheduler

    torch.manual_seed(0)
    _, model, optimizer, scheduler = setup()
    path = tmp_path / "ckpt.pth"
    torch.save({"model": model.state_dict(), "model_config": model.final_config, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": 0}, path)

    mgr, model2, optimizer2, scheduler2 = setup()
    resumed = load_checkpoint(path, model2, optimizer2, scheduler2, mgr, torch.device("cpu"),
                              load_weights_only=weights_only)[0]
    x = torch.randn(1, 1, 16, 16, 16) * 3
    resumed.train()
    train_out = _out(resumed, x)
    resumed.eval()
    assert torch.allclose(_out(resumed, x), train_out, atol=1e-5)
