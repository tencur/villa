"""Differentiable sparse chunk cache 3D grid_sample with gradients w.r.t. sample positions."""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load as _load_ext

_module = None


def _get_module():
    global _module
    if _module is None:
        src = str(Path(__file__).with_name("sparse_grid_sample_3d_u8_diff_kernel.cu"))
        _module = _load_ext(
            name="sparse_grid_sample_3d_u8_diff_ext",
            sources=[src],
            extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            extra_cuda_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            verbose=False,
        )
    return _module


class _SparseGridSample3DU8Diff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, chunk_table, C, grid, offset, inv_scale, pair_a, pair_b):
        mod = _get_module()
        out = mod.sparse_grid_sample_3d_u8_diff_fwd(chunk_table, C, grid, offset, inv_scale, pair_a, pair_b)
        ctx.save_for_backward(grid)
        ctx.chunk_table = chunk_table
        ctx.C = C
        ctx.offset = offset
        ctx.inv_scale = inv_scale
        ctx.pair_a = pair_a
        ctx.pair_b = pair_b
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grid, = ctx.saved_tensors
        mod = _get_module()
        grad_grid = mod.sparse_grid_sample_3d_u8_diff_bwd(
            ctx.chunk_table, ctx.C, grid, ctx.offset, ctx.inv_scale,
            grad_output.contiguous(), ctx.pair_a, ctx.pair_b,
        )
        return None, None, grad_grid, None, None, None, None


def sparse_grid_sample_3d_u8_diff(
    chunk_table: torch.Tensor,
    C: int,
    grid: torch.Tensor,
    offset: torch.Tensor,
    inv_scale: torch.Tensor,
    pair_a: int = -1,
    pair_b: int = -1,
) -> torch.Tensor:
    """Trilinear 3D grid_sample from sparse chunk cache, differentiable w.r.t. grid positions.

    Args:
        chunk_table: (cZ, cY, cX) int64 CUDA — device pointers (0 = empty)
        C: number of channels per chunk
        grid: (D, H, W, 3) float32 CUDA — fullres coordinates (x, y, z)
        offset: (3,) float32 CUDA — origin in fullres coords
        inv_scale: (3,) float32 CUDA — 1.0 / spacing per axis
        pair_a, pair_b: chunk channel indices of the hemisphere-encoded (nx, ny)
            normal pair, blended sign-aware (see normal_sampling.py); -1 = none

    Returns:
        (C, D, H, W) float32 CUDA — raw interpolated values (not decoded)
    """
    return _SparseGridSample3DU8Diff.apply(chunk_table, C, grid, offset, inv_scale, int(pair_a), int(pair_b))
