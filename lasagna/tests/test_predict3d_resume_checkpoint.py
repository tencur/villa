"""predict3d resumes an existing output tile by tile, so the output must come from the same checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lasagna.inference_provenance import (
    CheckpointMismatchError,
    check_resume_checkpoint,
    previous_checkpoint_sha256,
)


def _write_provenance(path: Path, sha: str) -> None:
    path.write_text(json.dumps({"artifact_kind": "lasagna", "checkpoint": {"sha256": sha}, "status": "completed"}))


def test_fresh_output_has_no_previous_checkpoint(tmp_path: Path) -> None:
    assert previous_checkpoint_sha256(tmp_path / "inference.json") is None
    assert check_resume_checkpoint(tmp_path / "inference.json", "a" * 64) is None


def test_same_checkpoint_resumes(tmp_path: Path) -> None:
    _write_provenance(tmp_path / "inference.json", "a" * 64)
    assert check_resume_checkpoint(tmp_path / "inference.json", "a" * 64) == "a" * 64


def test_different_checkpoint_is_refused(tmp_path: Path) -> None:
    _write_provenance(tmp_path / "inference.json", "a" * 64)
    with pytest.raises(CheckpointMismatchError, match="--allow-checkpoint-change"):
        check_resume_checkpoint(tmp_path / "inference.json", "b" * 64)


def test_different_checkpoint_allowed_explicitly(tmp_path: Path) -> None:
    _write_provenance(tmp_path / "inference.json", "a" * 64)
    assert check_resume_checkpoint(tmp_path / "inference.json", "b" * 64, allow_change=True) == "a" * 64


def test_unreadable_provenance_does_not_block(tmp_path: Path) -> None:
    (tmp_path / "inference.json").write_text("not json")
    assert check_resume_checkpoint(tmp_path / "inference.json", "a" * 64) is None
