"""Under DDP each rank must draw its semi-supervised batches from a disjoint share of the patches."""

from __future__ import annotations

from types import SimpleNamespace

from vesuvius.models.training.trainers.semi_supervised.two_stream_batch_sampler import (
    TwoStreamBatchSampler,
    shard_for_rank,
)


def test_shards_are_disjoint_and_cover_everything():
    idx = list(range(20))
    ranks = [SimpleNamespace(is_distributed=True, rank=r, world_size=3) for r in range(3)]
    shards = [shard_for_rank(idx, t) for t in ranks]
    assert sorted(i for s in shards for i in s) == idx
    assert all(set(a).isdisjoint(b) for i, a in enumerate(shards) for b in shards[i + 1:])


def test_single_process_keeps_all_indices():
    assert shard_for_rank(range(5), SimpleNamespace(is_distributed=False)) == [0, 1, 2, 3, 4]


def test_samplers_on_two_ranks_never_share_a_labeled_patch():
    labeled, unlabeled = list(range(40)), list(range(40, 60))
    seen = []
    for r in range(2):
        t = SimpleNamespace(is_distributed=True, rank=r, world_size=2)
        sampler = TwoStreamBatchSampler(shard_for_rank(labeled, t), shard_for_rank(unlabeled, t), 4, 2)
        seen.append({i for batch in sampler for i in batch if i < 40})
    assert seen[0].isdisjoint(seen[1])
