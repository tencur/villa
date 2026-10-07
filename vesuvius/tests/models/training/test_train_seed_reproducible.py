"""--seed must make vesuvius.train repeatable: model init, sampler order and worker seeds all come from it."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import SubsetRandomSampler

from vesuvius.models.training.train import BaseTrainer


def _trainer(seed):
    return BaseTrainer(mgr=SimpleNamespace(seed=seed), verbose=False)


def test_seed_everything_repeats_torch_draws():
    t = _trainer(42)
    t._seed_everything(); first = torch.randn(8)
    t._seed_everything(); second = torch.randn(8)
    assert torch.equal(first, second)
    _trainer(7)._seed_everything(); other = torch.randn(8)
    assert not torch.equal(first, other)


def test_sampler_order_follows_the_seed():
    order = lambda seed: list(SubsetRandomSampler(list(range(50)), generator=_trainer(seed)._seed_generator(1)))
    assert order(42) == order(42)
    assert order(42) != order(7)


def test_no_seed_means_no_generator():
    assert _trainer(None)._seed_generator(1) is None


def test_generators_differ_across_ranks():
    a = _trainer(42); a.rank = 0
    b = _trainer(42); b.rank = 1
    assert not torch.equal(torch.randn(4, generator=a._seed_generator(3)), torch.randn(4, generator=b._seed_generator(3)))


def test_training_seeds_before_anything_else(monkeypatch):
    """_initialize_training must call _seed_everything first (model init and samplers draw after it)."""

    class _Seeded(Exception):
        pass

    def stop(self):
        raise _Seeded()

    monkeypatch.setattr(BaseTrainer, "_seed_everything", stop)
    try:
        _trainer(42)._initialize_training()
    except _Seeded:
        return
    raise AssertionError("_initialize_training did not seed")


def test_seed_streams_do_not_collide_across_seeds_and_ranks():
    # With seed + offset arithmetic, seed 42's sampler generator equalled seed 43's global seed.
    draw = lambda g: torch.randn(4, generator=g)
    a, b = _trainer(42), _trainer(43)
    assert not torch.equal(draw(a._seed_generator(1)), draw(b._seed_generator(0)))
    r1 = _trainer(42); r1.rank = 1
    r0 = _trainer(1042); r0.rank = 0
    assert not torch.equal(draw(r1._seed_generator(3)), draw(r0._seed_generator(3)))


def test_train_val_split_leaves_each_rank_its_own_numpy_stream():
    """The split must not reset the global numpy RNG to the same value on every DDP rank."""
    import numpy as np

    def numpy_after_seeding(rank):
        t = _trainer(42); t.rank = rank
        t._seed_everything()
        np.random.RandomState(42).shuffle(list(range(10)))   # what the split does now
        return np.random.rand(4)

    assert not np.array_equal(numpy_after_seeding(0), numpy_after_seeding(1))
