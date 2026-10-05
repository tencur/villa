"""A write that fails after the last submit must not be lost."""

from __future__ import annotations

import pytest

from vesuvius.models.run.patch_writer import BoundedPatchWriter


class _FailingStore:
    def __init__(self, fail_at):
        self.fail_at = fail_at
        self.written = []

    def __setitem__(self, index, value):
        if index == self.fail_at:
            raise OSError(28, "No space left on device")
        self.written.append(index)


def test_failure_of_the_last_write_is_raised_when_the_writer_closes():
    store = _FailingStore(fail_at=1)
    with pytest.raises(OSError, match="No space left"):
        with BoundedPatchWriter(store, max_workers=2) as writer:
            writer.submit(0, "patch 0")
            writer.submit(1, "patch 1")
    assert store.written == [0]


def test_successful_writes_close_cleanly():
    store = _FailingStore(fail_at=None)
    with BoundedPatchWriter(store, max_workers=2) as writer:
        for index in range(5):
            writer.submit(index, f"patch {index}")
    assert sorted(store.written) == [0, 1, 2, 3, 4]


def test_an_exception_in_the_loop_is_not_replaced_by_a_write_error():
    store = _FailingStore(fail_at=0)
    with pytest.raises(KeyError):
        with BoundedPatchWriter(store, max_workers=1) as writer:
            writer.submit(0, "patch 0")
            raise KeyError("raised by the inference loop")
