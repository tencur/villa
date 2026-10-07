"""The three most recent epoch checkpoints must survive rotation, as the training docs describe."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from vesuvius.models.training.save_checkpoint import manage_checkpoint_history


def test_three_recent_checkpoints_are_kept_when_the_trainer_records_each_epoch(tmp_path: Path):
    history, best = deque(maxlen=3), []
    losses = [0.8, 0.85, 0.95, 0.99, 0.999]         # best = epochs 1-2, recent = epochs 3-5
    for epoch, loss in enumerate(losses, start=1):
        path = tmp_path / f"m_epoch{epoch}.pth"; path.write_bytes(b"x")
        history.append((epoch, str(path)))          # as BaseTrainer._on_epoch_end does
        history, best = manage_checkpoint_history(
            checkpoint_history=history, best_checkpoints=best, epoch=epoch, checkpoint_path=path,
            validation_loss=loss, checkpoint_dir=tmp_path, model_name="m", max_recent=3, max_best=2)
    kept = sorted(int(p.stem.split("epoch")[1]) for p in tmp_path.glob("m_epoch*.pth"))
    assert {3, 4, 5} <= set(kept), f"recent epochs 3-5 must be kept, kept {kept}"
    assert {1, 2} <= set(kept), f"best epochs (0.8, 0.85) must be kept, kept {kept}"
