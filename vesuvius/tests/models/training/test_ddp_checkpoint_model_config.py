"""Checkpoints written under DDP must carry model_config: the DDP wrapper hides the network's final_config."""

from __future__ import annotations

import socket
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from vesuvius.models.training.train import BaseTrainer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_final_config_is_read_through_the_ddp_wrapper():
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{_free_port()}", rank=0, world_size=1)
    try:
        net = torch.nn.Linear(2, 2)
        net.final_config = {"model_name": "t", "patch_size": (8, 8, 8)}
        wrapped = DDP(net)
        assert getattr(wrapped, "final_config", None) is None  # the wrapper does not forward attributes
        trainer = BaseTrainer(mgr=SimpleNamespace(), verbose=False)
        assert getattr(trainer._unwrap_model(wrapped), "final_config", None) == net.final_config
    finally:
        dist.destroy_process_group()
