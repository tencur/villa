"""Sign-aware sampling of hemisphere-encoded (nx, ny) normal codes (normal_sampling.py).

The predict3d writer stores (nx, ny) * sign(nz) as uint8 codes, so two neighbouring
voxels of one sheet can hold codes of opposite sign wherever the true normal is close
to the z = 0 plane.  Blending such codes per channel cancels the in-plane part and
FitData3D.normal_3d then tips the normal towards +z.  These tests pin down:

* agreeing corners: the sign-aware blend is the plain trilinear blend (incl. zero padding);
* opposite-hemisphere neighbours: the blend recovers the plane normal instead of +z;
* a smooth synthetic sheet with randomly flipped codes samples to coherent normals;
* autograd w.r.t. the sample position matches finite differences;
* FitData3D's dense torch path and the sparse caches are wired to the new sampler.
"""
from __future__ import annotations

import math
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fit_data
import normal_sampling as ns


def _encode(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Writer convention (normal_encoding.encode_normal_nxny_u8): store (nx, ny) * sign(nz)."""
	flip = torch.where(n[..., 2] < 0.0, -1.0, 1.0)
	nx = (n[..., 0] * flip * 127.0 + 128.0).round().clamp(0, 255).to(torch.uint8)
	ny = (n[..., 1] * flip * 127.0 + 128.0).round().clamp(0, 255).to(torch.uint8)
	return nx, ny


def _unit(v: torch.Tensor) -> torch.Tensor:
	return v / v.norm(dim=-1, keepdim=True)


def _angle_deg_unsigned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	"""Angle between lines (sign-free), degrees."""
	d = (a * b).sum(dim=-1).abs().clamp(max=1.0)
	return torch.rad2deg(torch.acos(d))


def _plain_trilinear(vol_u8: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
	"""F.grid_sample reference: trilinear, zero padding, align_corners=True; (Z,Y,X) uint8 -> (...)."""
	Z, Y, X = vol_u8.shape
	g = xyz.clone().float()
	g[..., 0] = g[..., 0] / max(1, X - 1) * 2 - 1
	g[..., 1] = g[..., 1] / max(1, Y - 1) * 2 - 1
	g[..., 2] = g[..., 2] / max(1, Z - 1) * 2 - 1
	out = F.grid_sample(
		vol_u8.float()[None, None], g.reshape(1, 1, 1, -1, 3),
		mode="bilinear", padding_mode="zeros", align_corners=True,
	)
	return out.reshape(xyz.shape[:-1])


def _smooth_agreeing_field(Z: int, Y: int, X: int) -> torch.Tensor:
	"""Smoothly varying unit normals with nz >= 0.5: no two neighbours disagree in sign."""
	z, y, x = torch.meshgrid(
		torch.arange(Z, dtype=torch.float32),
		torch.arange(Y, dtype=torch.float32),
		torch.arange(X, dtype=torch.float32),
		indexing="ij",
	)
	theta = 0.15 * x + 0.1 * y + 0.05 * z
	tilt = 0.4 + 0.3 * torch.sin(0.2 * x + 0.1 * z)  # in-plane magnitude, nz = sqrt(1 - tilt^2) >= 0.71
	return torch.stack([tilt * torch.cos(theta), tilt * torch.sin(theta), torch.sqrt(1 - tilt * tilt)], dim=-1)


class TestAgreeingCorners(unittest.TestCase):
	def test_matches_plain_trilinear_including_zero_padding(self) -> None:
		torch.manual_seed(0)
		n = _smooth_agreeing_field(6, 7, 8)
		nx, ny = _encode(n)
		# Sample positions inside and partly outside the volume: zero-padded corners (code 0,0)
		# are never flipped, so the blend must equal the plain one there too.
		xyz = torch.rand(500, 3) * torch.tensor([9.0, 8.0, 7.0]) - 1.0
		a_sa, b_sa = ns.sample_normal_codes(nx, ny, xyz, sign_aware=True)
		a_pl, b_pl = ns.sample_normal_codes(nx, ny, xyz, sign_aware=False)
		self.assertTrue(torch.allclose(a_sa, a_pl, atol=1e-3))
		self.assertTrue(torch.allclose(b_sa, b_pl, atol=1e-3))
		self.assertTrue(torch.allclose(a_sa, _plain_trilinear(nx, xyz), atol=1e-2))
		self.assertTrue(torch.allclose(b_sa, _plain_trilinear(ny, xyz), atol=1e-2))

	def test_round_to_u8_matches_rounded_plain_blend(self) -> None:
		torch.manual_seed(1)
		n = _smooth_agreeing_field(5, 5, 5)
		nx, ny = _encode(n)
		xyz = torch.rand(300, 3) * 4.0
		a_sa, _ = ns.sample_normal_codes(nx, ny, xyz, sign_aware=True, round_to_u8=True)
		a_pl, _ = ns.sample_normal_codes(nx, ny, xyz, sign_aware=False)
		ref = torch.floor(a_pl + 0.5).clamp(0, 255)
		# Rounding may differ only where the plain blend sits within float noise of a .5 boundary.
		frac = (a_pl + 0.5) - torch.floor(a_pl + 0.5)
		stable = (frac - 0.0).abs() > 1e-3
		self.assertTrue(torch.equal(a_sa[stable], ref[stable]))
		self.assertEqual(a_sa.dtype, torch.float32)
		self.assertTrue(bool((a_sa >= 0).all()) and bool((a_sa <= 255).all()))

	def test_exact_voxel_positions_return_stored_codes(self) -> None:
		n = _smooth_agreeing_field(4, 4, 4)
		nx, ny = _encode(n)
		z, y, x = torch.meshgrid(torch.arange(4), torch.arange(4), torch.arange(4), indexing="ij")
		xyz = torch.stack([x, y, z], dim=-1).float()
		a, b = ns.sample_normal_codes(nx, ny, xyz)
		self.assertTrue(torch.equal(a.round().to(torch.uint8), nx))
		self.assertTrue(torch.equal(b.round().to(torch.uint8), ny))


class TestOppositeHemisphereNeighbours(unittest.TestCase):
	"""The C41 defect: two voxels, same plane normal, stored with opposite hemisphere sign."""

	def setUp(self) -> None:
		theta = math.radians(35.0)
		nz = 0.04
		r = math.sqrt(1.0 - nz * nz)
		self.n_true = torch.tensor([r * math.cos(theta), r * math.sin(theta), nz])
		# Voxel x=0 saw nz = +0.04, voxel x=1 saw nz = -0.04: the writer flips (nx, ny) at x=1.
		field = torch.zeros(1, 1, 2, 3)
		field[0, 0, 0] = self.n_true
		field[0, 0, 1] = self.n_true * torch.tensor([1.0, 1.0, -1.0])
		self.nx, self.ny = _encode(field)
		self.assertNotEqual(int(self.nx[0, 0, 0]), int(self.nx[0, 0, 1]))  # codes really are mirrored
		self.mid = torch.tensor([[0.5, 0.0, 0.0]])

	def _sampled_normal(self, sign_aware: bool) -> torch.Tensor:
		a, b = ns.sample_normal_codes(self.nx, self.ny, self.mid, sign_aware=sign_aware)
		return ns.normal_from_components(ns.decode_codes(a), ns.decode_codes(b))[0]

	def test_plain_blend_cancels_to_plus_z(self) -> None:
		n_plain = self._sampled_normal(sign_aware=False)
		self.assertGreater(float(n_plain[2]), 0.95)  # the "normal" points along +z
		self.assertGreater(float(_angle_deg_unsigned(n_plain, self.n_true)), 60.0)

	def test_sign_aware_blend_recovers_plane_normal(self) -> None:
		n_sa = self._sampled_normal(sign_aware=True)
		self.assertLess(float(_angle_deg_unsigned(n_sa, self.n_true)), 3.0)
		self.assertGreaterEqual(float(n_sa[2]), 0.0)  # stays in the nz >= 0 convention

	def test_sign_aware_blend_is_continuous_across_the_seam(self) -> None:
		xs = torch.linspace(0.0, 1.0, 21)
		xyz = torch.stack([xs, torch.zeros_like(xs), torch.zeros_like(xs)], dim=-1)
		a, b = ns.sample_normal_codes(self.nx, self.ny, xyz, sign_aware=True)
		n = ns.normal_from_components(ns.decode_codes(a), ns.decode_codes(b))
		ang = _angle_deg_unsigned(n, self.n_true.expand_as(n))
		self.assertLess(float(ang.max()), 3.0)

	def test_reference_corner_is_the_heaviest_one(self) -> None:
		# Near x=1 the heavy corner is the flipped voxel: output must still be reported nz >= 0
		# and equal (up to quantisation) to the stored code of that voxel.
		a, b = ns.sample_normal_codes(self.nx, self.ny, torch.tensor([[0.9, 0.0, 0.0]]))
		n = ns.normal_from_components(ns.decode_codes(a), ns.decode_codes(b))[0]
		self.assertGreaterEqual(float(n[2]), 0.0)
		self.assertLess(float(_angle_deg_unsigned(n, self.n_true)), 3.0)


class TestSmoothSheetWithRandomFlips(unittest.TestCase):
	def _make(self, seed: int = 3, Z: int = 10, Y: int = 11, X: int = 12):
		g = torch.Generator().manual_seed(seed)
		z, y, x = torch.meshgrid(
			torch.arange(Z, dtype=torch.float32),
			torch.arange(Y, dtype=torch.float32),
			torch.arange(X, dtype=torch.float32),
			indexing="ij",
		)
		theta = 0.12 * x + 0.08 * y + 0.04 * z          # smoothly turning near-vertical sheet
		eps = 0.03                                      # |nz| of the sheet, tiny
		sgn = torch.where(torch.rand(Z, Y, X, generator=g) < 0.5, -1.0, 1.0)  # noisy nz sign per voxel
		n = torch.stack([torch.cos(theta), torch.sin(theta), eps * sgn], dim=-1)
		n = _unit(n)
		nx, ny = _encode(n)

		def true_normal(xyz: torch.Tensor) -> torch.Tensor:
			th = 0.12 * xyz[..., 0] + 0.08 * xyz[..., 1] + 0.04 * xyz[..., 2]
			return _unit(torch.stack([torch.cos(th), torch.sin(th), torch.full_like(th, eps)], dim=-1))

		xyz = torch.rand(4000, 3, generator=g) * torch.tensor([X - 1.0, Y - 1.0, Z - 1.0])
		return nx, ny, xyz, true_normal(xyz)

	def _angles(self, nx, ny, xyz, n_true, sign_aware: bool) -> torch.Tensor:
		a, b = ns.sample_normal_codes(nx, ny, xyz, sign_aware=sign_aware)
		n = ns.normal_from_components(ns.decode_codes(a), ns.decode_codes(b))
		return _angle_deg_unsigned(n, n_true)

	def test_sign_aware_sampling_is_coherent_where_plain_sampling_is_not(self) -> None:
		nx, ny, xyz, n_true = self._make()
		ang_sa = self._angles(nx, ny, xyz, n_true, sign_aware=True)
		ang_pl = self._angles(nx, ny, xyz, n_true, sign_aware=False)
		# Plain blending: most samples straddle a flip and the rebuilt normal tips towards +z.
		self.assertGreater(float((ang_pl > 30.0).float().mean()), 0.5)
		# Sign-aware: a coherent sheet.  (The residual comes from blending unit vectors that
		# differ by up to ~1.5 rad across a cell and re-deriving nz; see PR notes.)
		self.assertLess(float(ang_sa.median()), 5.0)
		self.assertLess(float((ang_sa > 30.0).float().mean()), 0.01)
		self.assertLess(float(ang_sa.max()), 30.0)

	def test_kill_switch_restores_plain_blend(self) -> None:
		with mock.patch.dict(os.environ, {"LASAGNA_SIGN_AWARE_NORMALS": "0"}):
			self.assertFalse(ns.sign_aware_enabled())
			self.assertEqual(ns.kernel_normal_pair(["grad_mag", "nx", "ny"]), (-1, -1))
		with mock.patch.dict(os.environ, {"LASAGNA_SIGN_AWARE_NORMALS": "1"}):
			self.assertEqual(ns.kernel_normal_pair(["grad_mag", "nx", "ny"]), (1, 2))
			self.assertEqual(ns.kernel_normal_pair(["cos"]), (-1, -1))
			self.assertEqual(ns.kernel_normal_pair(["nx"]), (-1, -1))


class TestAutograd(unittest.TestCase):
	def test_gradient_wrt_position_matches_finite_differences(self) -> None:
		torch.manual_seed(5)
		theta = math.radians(20.0)
		# A flipped-code checkerboard with a smooth underlying sheet, sampled strictly inside cells
		# whose sign pattern does not change in a small neighbourhood (the blend is smooth there).
		Z = Y = X = 5
		z, y, x = torch.meshgrid(*(torch.arange(5, dtype=torch.float32),) * 3, indexing="ij")
		th = theta + 0.1 * x + 0.05 * y
		sgn = torch.where(((x + y + z) % 2) == 0, 1.0, -1.0)
		n = _unit(torch.stack([torch.cos(th), torch.sin(th), 0.05 * sgn], dim=-1))
		nx, ny = _encode(n)
		xyz = torch.tensor([[1.3, 2.6, 1.7], [2.2, 1.4, 3.3]], dtype=torch.float64, requires_grad=True)
		a, b = ns.sample_normal_codes(nx, ny, xyz)
		loss = (a * torch.tensor([1.0, 2.0], dtype=torch.float64)).sum() + (b * 0.5).sum()
		grad, = torch.autograd.grad(loss, xyz)
		h = 1e-4
		for i in range(xyz.shape[0]):
			for k in range(3):
				e = torch.zeros_like(xyz)
				e[i, k] = h
				with torch.no_grad():
					ap, bp = ns.sample_normal_codes(nx, ny, xyz + e)
					am, bm = ns.sample_normal_codes(nx, ny, xyz - e)
				wa = torch.tensor([1.0, 2.0], dtype=torch.float64)
				lp = (ap * wa).sum() + (bp * 0.5).sum()
				lm = (am * wa).sum() + (bm * 0.5).sum()
				fd = float((lp - lm) / (2 * h))
				self.assertAlmostEqual(float(grad[i, k]), fd, delta=1e-2 * max(1.0, abs(fd)))


def _fit_data_from_codes(nx: torch.Tensor, ny: torch.Tensor) -> fit_data.FitData3D:
	Z, Y, X = nx.shape
	return fit_data.FitData3D(
		cos=None,
		grad_mag=torch.full((1, 1, Z, Y, X), 200, dtype=torch.uint8),
		nx=nx[None, None].clone(),
		ny=ny[None, None].clone(),
		pred_dt=None,
		corr_points=None,
		winding_volume=None,
		origin_fullres=(10.0, 20.0, 30.0),
		spacing=(2.0, 2.0, 2.0),
		cuda_gridsample=False,
	)


class TestFitDataDensePath(unittest.TestCase):
	def setUp(self) -> None:
		theta = math.radians(35.0)
		nz = 0.04
		r = math.sqrt(1.0 - nz * nz)
		self.n_true = torch.tensor([r * math.cos(theta), r * math.sin(theta), nz])
		field = self.n_true.expand(3, 3, 3, 3).clone()
		field[:, :, 1::2, 2] *= -1.0  # every other x column is stored in the opposite hemisphere
		nx, ny = _encode(field)
		self.data = _fit_data_from_codes(nx, ny)
		# fullres position of local voxel-space (0.5, 1.0, 1.0): origin + spacing * local
		self.xyz = torch.tensor([[[[10.0 + 2.0 * 0.5, 20.0 + 2.0, 30.0 + 2.0]]]])  # (1,1,1,3)

	def test_torch_path_samples_sign_aware(self) -> None:
		sampled = self.data.grid_sample_fullres(self.xyz)
		n = sampled.normal_3d.reshape(3)
		self.assertLess(float(_angle_deg_unsigned(n, self.n_true)), 3.0)
		self.assertEqual(tuple(sampled.nx.shape), (1, 1, 1, 1, 1))
		self.assertAlmostEqual(float(sampled.grad_mag.reshape(())), 200.0 / 255.0, places=4)

	def test_torch_path_kill_switch_reproduces_old_tilt(self) -> None:
		with mock.patch.dict(os.environ, {"LASAGNA_SIGN_AWARE_NORMALS": "0"}):
			sampled = self.data.grid_sample_fullres(self.xyz)
		n = sampled.normal_3d.reshape(3)
		self.assertGreater(float(_angle_deg_unsigned(n, self.n_true)), 60.0)

	def test_channel_selection_is_respected(self) -> None:
		s = self.data.grid_sample_fullres(self.xyz, channels={"nx"})
		self.assertIsNotNone(s.nx)
		self.assertIsNone(s.ny)
		self.assertIsNone(s.grad_mag)
		s = self.data.grid_sample_fullres(self.xyz, channels={"grad_mag"})
		self.assertIsNone(s.nx)
		self.assertIsNone(s.ny)
		self.assertIsNotNone(s.grad_mag)

	def test_agreeing_volume_matches_previous_torch_path(self) -> None:
		n = _smooth_agreeing_field(4, 5, 6)
		nx, ny = _encode(n)
		data = _fit_data_from_codes(nx, ny)
		torch.manual_seed(7)
		local = torch.rand(2, 3, 4, 3) * torch.tensor([5.0, 4.0, 3.0])
		xyz = torch.tensor([10.0, 20.0, 30.0]) + 2.0 * local
		new = data.grid_sample_fullres(xyz)
		with mock.patch.dict(os.environ, {"LASAGNA_SIGN_AWARE_NORMALS": "0"}):
			old = data.grid_sample_fullres(xyz)
		self.assertTrue(torch.allclose(new.nx, old.nx, atol=1e-4))
		self.assertTrue(torch.allclose(new.ny, old.ny, atol=1e-4))


class TestSparseCacheWiring(unittest.TestCase):
	"""The CUDA kernels cannot run here; check the caches hand them the (nx, ny) pair."""

	def _fake_cache(self, cls, channels: list[str]):
		cache = cls.__new__(cls)
		cache.channels = channels
		cache.n_channels = len(channels)
		cache.chunk_table = torch.zeros(1, 1, 1, dtype=torch.int64)
		cache.device = torch.device("cpu")
		return cache

	def _run(self, cls, channels: list[str], diff: bool):
		calls = []

		def fake_kernel(chunk_table, C, grid, offset, inv_scale, pair_a=-1, pair_b=-1):
			calls.append((C, pair_a, pair_b))
			return torch.zeros(C, *grid.shape[:-1])

		mods = {
			"sparse_grid_sample_3d_u8": types.SimpleNamespace(sparse_grid_sample_3d_u8=fake_kernel),
			"sparse_grid_sample_3d_u8_diff": types.SimpleNamespace(sparse_grid_sample_3d_u8_diff=fake_kernel),
		}
		cache = self._fake_cache(cls, channels)
		grid = torch.zeros(1, 1, 2, 3)
		with mock.patch.dict(sys.modules, mods), mock.patch.dict(os.environ, {"LASAGNA_CHECK_SPARSE_CACHE": "0"}):
			out = cache.grid_sample(grid, torch.zeros(3), torch.ones(3), diff=diff)
		self.assertEqual(tuple(out.shape), (len(channels), 1, 1, 2))
		return calls

	def test_python_zarr_cache_passes_pair_for_both_kernels(self) -> None:
		import sparse_cache
		for diff in (False, True):
			self.assertEqual(self._run(sparse_cache.SparseChunkGroupCache, ["grad_mag", "nx", "ny"], diff), [(3, 1, 2)])
			self.assertEqual(self._run(sparse_cache.SparseChunkGroupCache, ["cos"], diff), [(1, -1, -1)])

	def test_pair_disabled_by_kill_switch(self) -> None:
		import sparse_cache
		with mock.patch.dict(os.environ, {"LASAGNA_SIGN_AWARE_NORMALS": "0"}):
			self.assertEqual(self._run(sparse_cache.SparseChunkGroupCache, ["nx", "ny"], False), [(2, -1, -1)])

	def test_tensorstore_cache_passes_pair(self) -> None:
		try:
			import sparse_tensorstore_cache
		except ImportError as exc:  # tensorstore is a vesuvius dependency, not always installed
			self.skipTest(f"tensorstore not importable: {exc}")
		cls = sparse_tensorstore_cache.TensorStoreSparseChunkGroupCache
		self.assertEqual(self._run(cls, ["grad_mag", "nx", "ny"], False), [(3, 1, 2)])


if __name__ == "__main__":
	unittest.main()
