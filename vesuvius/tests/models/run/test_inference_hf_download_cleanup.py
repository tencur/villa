"""vesuvius.predict must not leave the downloaded copy of an hf:// model behind."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

import vesuvius.models.run.inference as inference
from vesuvius.models.run.inference import Inferer


class _StopAfterModelInfo(Exception):
    pass


def test_hf_model_download_directory_is_removed_after_loading(tmp_path, monkeypatch):
    download = Path(tempfile.mkdtemp(prefix="vesuvius_hf_model_", dir=tmp_path))
    (download / "fold_0").mkdir()
    (download / "fold_0" / "checkpoint_final.pth").write_bytes(b"weights")

    def fake_loader(**kwargs):
        assert kwargs["hf_model_path"] == "scrollprize/surface_recto"
        return {"temp_dir": str(download)}

    def stop(*args, **kwargs):
        # Called once the hf:// branch is done; the rest of loading is not under test.
        raise _StopAfterModelInfo

    monkeypatch.setattr(inference, "load_model_for_inference", fake_loader)
    monkeypatch.setattr(inference, "_resolve_model_path", lambda path, *a, **k: path)
    inferer = Inferer.__new__(Inferer)
    inferer.model_path = "hf://scrollprize/surface_recto"
    inferer.model_cache_dir = str(tmp_path / "cache")
    inferer.verbose = False
    inferer.device = torch.device("cpu")
    monkeypatch.setattr(
        Inferer, "_load_train_py_model", lambda self, path: pytest.fail("not a train.py model")
    )
    monkeypatch.setattr(inference, "get_model_normalization_info", stop, raising=False)

    try:
        inferer._load_model()
    except (_StopAfterModelInfo, KeyError, AttributeError, TypeError):
        pass  # the fake model_info is not a full model; only the cleanup is checked

    assert not download.exists()
