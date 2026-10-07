"""vesuvius.compute_st must integrate (smooth) the tensor components by default.

Without the second smoothing every voxel stores g g^T, which is rank 1: its two smaller eigenvectors - written as
first_component / second_component - are arbitrary, and FA/linearity confidence saturates.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from vesuvius.structure_tensor import run_create_st


def _parse(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compute_st", *argv])
    return run_create_st.parse_arguments()


def test_smoothing_is_on_by_default(monkeypatch):
    assert _parse(["--input_dir", "in.zarr", "--output_dir", "out"], monkeypatch).smooth_components is True


def test_it_can_be_turned_off(monkeypatch):
    args = _parse(["--input_dir", "in.zarr", "--output_dir", "out", "--no-smooth-components"], monkeypatch)
    assert args.smooth_components is False


def test_unsmoothed_tensor_is_rank_one_and_smoothed_is_not():
    from vesuvius.image_proc.geometry.structure_tensor import StructureTensorComputer
    rng = np.random.default_rng(0)
    vol = torch.from_numpy(rng.normal(size=(1, 1, 24, 24, 24)).astype(np.float32))
    comp = StructureTensorComputer(sigma=1.0, device="cpu")
    for smooth, check in ((False, lambda r: r < 1e-4), (True, lambda r: r > 1e-2)):
        J = comp.compute(vol, sigma=1.0, component_sigma=1.0 if smooth else None,
                         smooth_components=smooth, device="cpu")[0, :, 8:16, 8:16, 8:16]
        jzz, jzy, jzx, jyy, jyx, jxx = J.reshape(6, -1).double()
        M = torch.stack([torch.stack([jzz, jzy, jzx], -1), torch.stack([jzy, jyy, jyx], -1),
                         torch.stack([jzx, jyx, jxx], -1)], -2)
        w = torch.linalg.eigvalsh(M)
        ratio = float(torch.median(w[:, 1].abs() / w[:, 2].clamp_min(1e-12)))
        assert check(ratio), (smooth, ratio)



class _Parsed(Exception):
    pass


def _create_st_args(argv, monkeypatch):
    """Parse create_st's own CLI without running it."""
    import argparse
    from vesuvius.structure_tensor import create_st
    original = argparse.ArgumentParser.parse_args

    def capture(self, *a, **k):
        raise _Parsed(original(self, *a, **k))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(sys, "argv", ["create_st", *argv])
    try:
        create_st.main()
    except _Parsed as parsed:
        return parsed.args[0]
    raise AssertionError("create_st.main() did not parse its arguments")


def test_create_st_cli_integrates_by_default(monkeypatch):
    assert _create_st_args(["--input_dir", "in.zarr", "--output_dir", "out"], monkeypatch).smooth_components is True
    off = _create_st_args(["--input_dir", "in.zarr", "--output_dir", "out", "--no-smooth-components"], monkeypatch)
    assert off.smooth_components is False


def test_run_create_st_forwards_the_choice_to_create_st(monkeypatch):
    import subprocess
    seen = []

    class _Proc:
        returncode = 0
        def __init__(self, cmd, *a, **k): seen.append(cmd)
        def communicate(self, *a, **k): return ("", "")
        def wait(self, *a, **k): return 0
        stdout = iter(())

    monkeypatch.setattr(subprocess, "Popen", _Proc)
    for flag, expected in (([], "--smooth-components"), (["--no-smooth-components"], "--no-smooth-components")):
        seen.clear()
        args = _parse(["--input_dir", "in.zarr", "--output_dir", "out", *flag], monkeypatch)
        args.num_parts = 1  # set by run_create_st.main() before it launches the parts
        try:
            run_create_st.run_structure_tensor_part(args, 0, None, "out.zarr")
        except Exception:
            pass
        assert seen and expected in seen[0], seen


def test_python_api_integrates_by_default_too():
    import inspect
    from vesuvius.structure_tensor.create_st import StructureTensorInferer

    assert inspect.signature(StructureTensorInferer.__init__).parameters["smooth_components"].default is True
