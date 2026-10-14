"""pred-dt GPU EDT stage (_compute_pred_dt_slab): outside the predicted surface the encoded value must be
128 - clip(round(distance to the surface), 1, 48). The reader processes hand the GPU loop a uint8 0/1 mask,
so the outside EDT must run on `mask == 0`; `~mask` on uint8 is a bitwise NOT (255/254, all non-zero)."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
from scipy import ndimage
import zarr

import preprocess_cos_omezarr as pco


class _Stream:
	"""CPU stand-in for a CuPy stream: work runs synchronously."""

	def __init__(self, *args, **kwargs):
		pass

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		return False

	def record(self):
		return None

	def wait_event(self, _event):
		pass

	def synchronize(self):
		pass


class _HostArray(np.ndarray):
	def get(self, out=None):
		if out is None:
			return np.asarray(self)
		out[...] = self
		return out


def _spawn_workers(monkeypatch):
	"""Spawned reader/writer processes: forking after tensorstore's threads have
	started (as earlier tests in the session do) aborts the child."""
	import multiprocessing

	ctx = multiprocessing.get_context("spawn")
	for name in ("Process", "Queue", "Event"):
		monkeypatch.setattr(multiprocessing, name, getattr(ctx, name))


def _cpu_cupy(monkeypatch, encoded):
	"""Run the GPU EDT stage on NumPy/SciPy (same calls, same dtypes); record each chunk's
	input mask and encoded result as the loop produces them."""

	def where(cond, a, b):
		out = np.where(cond, a, b).view(_HostArray)
		encoded.append((np.array(cond), np.array(out)))
		return out

	cp = types.SimpleNamespace(
		cuda=types.SimpleNamespace(
			Device=lambda: types.SimpleNamespace(mem_info=(64 << 20, 64 << 20)),
			Stream=_Stream,
			get_current_stream=lambda: _Stream(),
		),
		asarray=lambda a: np.array(a).view(_HostArray),
		asnumpy=np.asarray,
		clip=np.clip,
		round=np.round,
		where=where,
		uint8=np.uint8,
		get_default_memory_pool=lambda: types.SimpleNamespace(free_all_blocks=lambda: None),
	)
	cnd = types.SimpleNamespace(distance_transform_edt=ndimage.distance_transform_edt)
	monkeypatch.setattr(pco, "cp", cp, raising=False)
	monkeypatch.setattr(pco, "cnd", cnd, raising=False)
	monkeypatch.setattr(pco, "_HAS_CUPY", True)


def test_pred_dt_outside_encodes_distance_to_surface(tmp_path: Path, monkeypatch):
	encoded = []
	_cpu_cupy(monkeypatch, encoded)
	_spawn_workers(monkeypatch)
	shape = (64, 64, 64)
	pred_path = tmp_path / "pred.zarr"
	pred = zarr.open_array(
		str(pred_path), mode="w", shape=shape, chunks=(32, 32, 32), dtype="uint8",
		zarr_format=2, dimension_separator="/",
	)
	sheet = np.zeros(shape, dtype=np.uint8)
	sheet[:, 30:34, :] = 255  # a 4-voxel surface sheet across y
	pred[:] = sheet
	out_path = tmp_path / "vol_pred_dt.ome.zarr"
	out_level = pco._open_or_create_omezarr(str(out_path), shape, 0, 1, 16, "pred_dt")["0"]

	pco._compute_pred_dt_slab(
		pred_zarr=pco.open_zarr(str(pred_path)),
		pred_path=str(pred_path),
		output_level_arr=out_level,
		output_omezarr_path=str(out_path),
		output_level_key="0",
		pred_z0=0, pred_z1=64, pred_y0=0, pred_y1=64, pred_x0=0, pred_x1=64,
		out_z0=0, out_y0=0, out_x0=0,
		scaledown=1, overlap=8, ome_chunk=16, n_levels=0, read_workers=1,
	)

	assert encoded, "the GPU EDT stage processed no chunk"
	for mask, enc in encoded:
		surface = mask > 0
		assert surface.any() and not surface.all()
		dist = ndimage.distance_transform_edt(~surface)
		expected = (128 - np.clip(np.round(dist), 1, 48)).astype(np.uint8)
		outside = ~surface
		wrong = int((enc[outside] != expected[outside]).sum())
		assert wrong == 0, f"{wrong} of {int(outside.sum())} outside voxels do not encode the distance to the surface"
		# the voxel right next to the sheet is 1 voxel away: 128 - 1 = 127
		next_to_sheet = outside & (ndimage.binary_dilation(surface) & ~surface)
		assert np.all(enc[next_to_sheet] == 127)
