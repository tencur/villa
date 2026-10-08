"""fiber_trace_3d.infer must run at the precision it resolves and records, not the checkpoint's training precision."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from vesuvius.neural_tracing.fiber_trace_3d import infer
from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import _config_from_checkpoint


def test_checkpoint_training_section_overrides_runtime_precision(tmp_path):
    # documents the mechanism the fix works around
    ckpt = tmp_path / "ck.pt"
    torch.save({"config": {"training": {"mixed_precision": "bf16"}}}, ckpt)
    merged = _config_from_checkpoint({"training": {"mixed_precision": "off"}}, ckpt)
    assert merged["training"]["mixed_precision"] == "bf16"


def test_apply_inference_precision_sets_the_resolved_mode():
    adapter = SimpleNamespace(config={"training": {"mixed_precision": "bf16", "lr": 1.0}})
    infer.apply_inference_precision(adapter, "off")
    assert adapter.config["training"] == {"mixed_precision": "off", "lr": 1.0}


def test_infer_applies_it_after_building_the_adapter():
    source = inspect.getsource(infer)
    assert "apply_inference_precision(predict_adapter, precision_mode)" in source
