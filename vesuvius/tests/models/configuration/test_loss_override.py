"""`vesuvius.train --loss` must change the losses the trainer builds, which come from each target's `losses` list."""

from __future__ import annotations

from types import SimpleNamespace

from vesuvius.models.configuration.config_utils import configure_targets


def _mgr():
    return SimpleNamespace(
        targets={"surface": {"out_channels": 1, "activation": "none",
                             "losses": [{"name": "BCEWithLogitsLoss", "weight": 1.0}]}},
        auxiliary_tasks={}, data_format="zarr", data_path=None, verbose=False,
    )


def test_loss_override_replaces_the_losses_list():
    mgr = _mgr()
    configure_targets(mgr, ["SoftDiceLoss"])
    assert [l["name"] for l in mgr.targets["surface"]["losses"]] == ["SoftDiceLoss"]
    assert mgr.targets["surface"]["loss_fn"] == "SoftDiceLoss"


def test_no_override_keeps_yaml_losses():
    mgr = _mgr()
    configure_targets(mgr, None)
    assert [l["name"] for l in mgr.targets["surface"]["losses"]] == ["BCEWithLogitsLoss"]
