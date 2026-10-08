"""Sign-aware trilinear sampling of Lasagna's hemisphere-encoded (nx, ny) normal codes.

The predict3d writer stores ``(nx, ny) * sign(nz)`` as uint8 codes
(``normal_encoding.encode_normal_nxny_u8``), so two neighbouring voxels that
describe the same sheet can hold codes of opposite sign whenever the true normal
lies close to the z = 0 plane.  Plain trilinear interpolation of such codes
cancels the in-plane component, and the normal rebuilt by
``fit_data.FitData3D.normal_3d`` (``nz = sqrt(1 - nx^2 - ny^2)``) tips towards +z.

This module is the device-agnostic reference implementation of the sign-aware
blend.  The sparse CUDA kernels (``sparse_grid_sample_3d_u8*_kernel.cu`` via
``sparse_normal_pair.cuh``) mirror it step by step; keep the two in sync.

1. Gather the 8 trilinear corner codes of both channels.  Out-of-volume corners
   read as code 0, exactly like the dense CUDA kernels and ``F.grid_sample`` with
   zero padding.
2. Decode each corner to a unit vector ``(nx, ny, nz = sqrt(max(0, 1 - nx^2 - ny^2)))``.
3. Take the corner with the largest trilinear weight (the first one on ties) as
   the reference and mark every corner whose dot product with it is negative as
   flipped.  A corner with code (0, 0) (zero padding; no valid normal encodes to it)
   is never the reference and never flipped.
4. Blend the codes with the ordinary trilinear weights, using the mirrored code
   ``256 - c`` for flipped corners (``(256 - c - 128) / 127 == -(c - 128) / 127``).
   If the blended nz is negative, mirror the result back into the nz >= 0
   hemisphere the rest of the fit expects.

When all 8 corners already agree, steps 3-4 are the identity and the result is
the ordinary trilinear blend of the codes, in the same summation order as the
existing kernels.
"""
from __future__ import annotations

import os

import torch

# Corner order shared with the CUDA kernels: k = dx + 2*dy + 4*dz.
_CORNERS: tuple[tuple[int, int, int], ...] = (
	(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
	(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)


def sign_aware_enabled() -> bool:
	"""Kill switch for A/B comparisons: ``LASAGNA_SIGN_AWARE_NORMALS=0`` restores plain blending."""
	return os.environ.get("LASAGNA_SIGN_AWARE_NORMALS", "1") != "0"


def kernel_normal_pair(channels: list[str]) -> tuple[int, int]:
	"""(pair_a, pair_b) kernel arguments for a chunk group's channel list: the indices of
	"nx" and "ny" when both are present and sign-aware sampling is enabled, else (-1, -1)."""
	if sign_aware_enabled() and "nx" in channels and "ny" in channels:
		return channels.index("nx"), channels.index("ny")
	return -1, -1


def decode_codes(codes: torch.Tensor) -> torch.Tensor:
	"""uint8-range code (as float) -> component in [-1, 1]."""
	return (codes - 128.0) / 127.0


def normal_from_components(nx: torch.Tensor, ny: torch.Tensor) -> torch.Tensor:
	"""Rebuild the nz >= 0 unit normal from decoded (nx, ny); mirrors ``FitData3D.normal_3d``."""
	nz = torch.sqrt(torch.clamp(1.0 - nx * nx - ny * ny, min=1e-8))
	n = torch.stack([nx, ny, nz], dim=-1)
	return n / (n.norm(dim=-1, keepdim=True) + 1e-8)


def trilinear_corners(xyz_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Floor indices and the 8 trilinear weights of sample positions.

	xyz_local: (..., 3) float, (x, y, z) in voxel index space.
	Returns (idx0 (..., 3) int64, weights (..., 8) float) with weights in ``_CORNERS`` order.
	The weights keep autograd history w.r.t. ``xyz_local``.
	"""
	x0 = torch.floor(xyz_local[..., 0])
	y0 = torch.floor(xyz_local[..., 1])
	z0 = torch.floor(xyz_local[..., 2])
	fx = xyz_local[..., 0] - x0
	fy = xyz_local[..., 1] - y0
	fz = xyz_local[..., 2] - z0
	ws = []
	for dx, dy, dz in _CORNERS:
		wx = fx if dx else (1.0 - fx)
		wy = fy if dy else (1.0 - fy)
		wz = fz if dz else (1.0 - fz)
		ws.append(wx * wy * wz)
	idx0 = torch.stack([x0, y0, z0], dim=-1).detach().long()
	return idx0, torch.stack(ws, dim=-1)


def gather_corner_codes(vol_u8: torch.Tensor, idx0: torch.Tensor) -> torch.Tensor:
	"""Read the 8 corner codes of a (Z, Y, X) uint8 volume as float32; out-of-volume corners are 0."""
	Z, Y, X = (int(v) for v in vol_u8.shape)
	flat = vol_u8.reshape(-1)
	out = []
	for dx, dy, dz in _CORNERS:
		ix = idx0[..., 0] + dx
		iy = idx0[..., 1] + dy
		iz = idx0[..., 2] + dz
		ok = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
		lin = (iz.clamp(0, Z - 1) * Y + iy.clamp(0, Y - 1)) * X + ix.clamp(0, X - 1)
		v = flat[lin].to(torch.float32)
		out.append(torch.where(ok, v, torch.zeros_like(v)))
	return torch.stack(out, dim=-1)


def align_corner_signs(
	codes_nx: torch.Tensor,
	codes_ny: torch.Tensor,
	weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Per-corner sign (+1 keep / -1 flip) relative to the heaviest corner, and the final hemisphere flip.

	All inputs are (..., 8).  Returns (sign (..., 8), flip (..., 1)), both float with values +-1.
	No gradient flows through the result (it is piecewise constant in the sample position).
	"""
	with torch.no_grad():
		ax = decode_codes(codes_nx)
		ay = decode_codes(codes_ny)
		az = torch.sqrt(torch.clamp(1.0 - ax * ax - ay * ay, min=0.0))
		# Code (0, 0) is never written by the encoder (it would decode to a vector of norm > 1);
		# it marks zero padding / unloaded data.  It is never the reference and never flipped, so
		# agreeing corners next to padding blend exactly as before.
		empty = (codes_nx == 0.0) & (codes_ny == 0.0)
		# Reference: heaviest non-empty corner, first maximum on ties (same as the kernel loop).
		ref = weights.detach().masked_fill(empty, -1.0).argmax(dim=-1, keepdim=True)
		rx = torch.gather(ax, -1, ref)
		ry = torch.gather(ay, -1, ref)
		rz = torch.gather(az, -1, ref)
		dot = ax * rx + ay * ry + az * rz
		one = torch.ones_like(dot)
		sign = torch.where((dot >= 0.0) | empty, one, -one)
		sz = (weights.detach() * sign * az).sum(dim=-1, keepdim=True)
		flip = torch.where(sz < 0.0, -one[..., :1], one[..., :1])
	return sign, flip


def blend_codes(
	codes: torch.Tensor,
	weights: torch.Tensor,
	sign: torch.Tensor,
	flip: torch.Tensor,
) -> torch.Tensor:
	"""Trilinear blend of (possibly mirrored) codes; (..., 8) -> (...)."""
	c = torch.where(sign > 0.0, codes, 256.0 - codes)
	val = (weights * c).sum(dim=-1)
	return torch.where(flip[..., 0] < 0.0, 256.0 - val, val)


def sample_normal_codes(
	nx_u8: torch.Tensor,
	ny_u8: torch.Tensor,
	xyz_local: torch.Tensor,
	*,
	sign_aware: bool = True,
	round_to_u8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Trilinearly sample the (nx, ny) code volumes at voxel-space positions.

	nx_u8, ny_u8: (Z, Y, X) uint8 volumes (same shape, same device).
	xyz_local: (..., 3) float, (x, y, z) in voxel index space of those volumes.
	sign_aware: align the 8 corners' hemisphere signs before blending (see module doc);
	    False gives the plain code blend the fit used before.
	round_to_u8: round and clamp the blended codes to [0, 255] like the non-differentiable
	    uint8 kernels do.

	Returns (nx_code, ny_code), float32 tensors shaped like ``xyz_local[..., 0]``, still in code
	space (decode with ``decode_codes``).  Differentiable w.r.t. ``xyz_local``.
	"""
	if nx_u8.shape != ny_u8.shape:
		raise ValueError(f"normal channel shape mismatch: nx={tuple(nx_u8.shape)} ny={tuple(ny_u8.shape)}")
	idx0, weights = trilinear_corners(xyz_local)
	ca = gather_corner_codes(nx_u8, idx0)
	cb = gather_corner_codes(ny_u8, idx0)
	if sign_aware:
		sign, flip = align_corner_signs(ca, cb, weights)
	else:
		sign = torch.ones_like(ca)
		flip = torch.ones_like(ca[..., :1])
	out_a = blend_codes(ca, weights, sign, flip)
	out_b = blend_codes(cb, weights, sign, flip)
	if round_to_u8:
		out_a = torch.floor(out_a + 0.5).clamp(0.0, 255.0)
		out_b = torch.floor(out_b + 0.5).clamp(0.0, 255.0)
	return out_a, out_b


__all__ = [
	"align_corner_signs",
	"blend_codes",
	"decode_codes",
	"gather_corner_codes",
	"kernel_normal_pair",
	"normal_from_components",
	"sample_normal_codes",
	"sign_aware_enabled",
	"trilinear_corners",
]
