"""vesuvius.predict must store logits for train.py checkpoints whose targets configure an output activation.

NetworkFromConfig applies each target's ``activation`` (sigmoid/softmax) in eval mode. The inference store is
documented as logits and finalize_outputs applies sigmoid/softmax itself, so the inference-side model must not.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.models.build.build_network_from_config import NetworkFromConfig, disable_eval_activations
from vesuvius.models.run.finalize_outputs import FinalizeConfig, apply_finalization


def _tiny_network(targets: dict) -> NetworkFromConfig:
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
    mgr = SimpleNamespace(
        model_config=model_config,
        targets=targets,
        train_patch_size=(16, 16, 16),
        train_batch_size=1,
        in_channels=1,
        autoconfigure=False,
        enable_deep_supervision=False,
        model_name="Model",
        spacing=[1, 1, 1],
    )
    torch.manual_seed(0)
    return NetworkFromConfig(mgr)


def _eval_output(model, x):
    model.eval()
    with torch.inference_mode():
        out = model(x)
    return out["ink"] if isinstance(out, dict) else out


@pytest.mark.parametrize("activation,out_channels", [("sigmoid", 1), ("softmax", 2)])
def test_eval_output_is_logits_after_disable(activation, out_channels):
    model = _tiny_network({"ink": {"out_channels": out_channels, "activation": activation}})
    x = torch.randn(1, 1, 16, 16, 16) * 3
    activated = _eval_output(model, x)
    disabled = disable_eval_activations(model)
    logits = _eval_output(model, x)

    assert disabled == [("ink", "Sigmoid" if activation == "sigmoid" else "Softmax")]
    expected = torch.sigmoid(logits) if activation == "sigmoid" else torch.softmax(logits, dim=1)
    assert torch.allclose(activated, expected, atol=1e-5)
    assert logits.min() < 0, "logits should not be confined to a probability range"


def test_disable_is_a_no_op_without_activations():
    model = _tiny_network({"ink": {"out_channels": 1, "activation": "none"}})
    assert disable_eval_activations(model) == []


def test_finalize_threshold_matches_probabilities_only_for_logits():
    model = _tiny_network({"ink": {"out_channels": 1, "activation": "sigmoid"}})
    x = torch.randn(1, 1, 16, 16, 16) * 3
    probabilities = _eval_output(model, x)[0].numpy()
    disable_eval_activations(model)
    logits = _eval_output(model, x)[0].numpy()

    truth = (probabilities > 0.5).astype(np.uint8) * 255
    from_logits, _ = apply_finalization(logits.astype(np.float32), 1, FinalizeConfig("binary", 0.5))
    from_probabilities, _ = apply_finalization(probabilities.astype(np.float32), 1, FinalizeConfig("binary", 0.5))
    assert np.array_equal(from_logits, truth)
    assert not np.array_equal(from_probabilities, truth)


def _saved_sigmoid_checkpoint(tmp_path):
    torch.manual_seed(0)
    net = _tiny_network({"ink": {"out_channels": 1, "activation": "sigmoid"}})
    path = tmp_path / "Model_epoch1.pth"
    torch.save({"model": net.state_dict(), "model_config": dict(net.final_config)}, path)
    x = torch.randn(1, 1, 16, 16, 16) * 3
    net.eval()
    with torch.no_grad():
        probabilities = net(x)["ink"]
    return path, x, probabilities


@pytest.mark.parametrize("loader", ["inferer", "load_model_from_checkpoint"])
def test_checkpoint_loaders_return_logits(tmp_path, loader):
    """Both paths vesuvius.predict uses to rebuild a train.py checkpoint must return logits."""
    path, x, probabilities = _saved_sigmoid_checkpoint(tmp_path)
    if loader == "inferer":
        from vesuvius.models.run.inference import Inferer

        inferer = Inferer.__new__(Inferer)
        inferer.verbose = False
        inferer.device = torch.device("cpu")
        model = inferer._load_train_py_model(path)["network"]
    else:
        from vesuvius.utils.models.load_nnunet_model import load_model_from_checkpoint

        model, _ = load_model_from_checkpoint(str(path), device="cpu")
    model.eval()
    with torch.no_grad():
        out = model(x)["ink"]
    assert torch.allclose(torch.sigmoid(out), probabilities, atol=1e-5)


def test_disable_looks_through_wrappers():
    model = _tiny_network({"ink": {"out_channels": 1, "activation": "sigmoid"}})

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    assert disable_eval_activations(Wrapper(model)) == [("ink", "Sigmoid")]
    assert model.task_activations["ink"] is None
