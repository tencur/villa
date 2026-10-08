"""Mirror / rotate TTA must transform 3-vector input fields (direction priors) with the volume.

The copy model decides which output slot is the front and which the back neighbour from the direction-prior
vectors in its input. Flipping the volume without flipping those vectors makes half of the mirror variants
point the prior at the opposite neighbour, so their aligned outputs have the slots swapped and the merge
lands between the two.
"""

from __future__ import annotations

import torch

from vesuvius.neural_tracing.inference.displacement_tta import run_model_tta


class _SlotContractModel(torch.nn.Module):
    """Stand-in obeying the training contract: slot A is the sheet on the +prior side, slot B the other one."""

    def forward(self, x):
        cond = x[:, 1] > 0                                   # (B, D, H, W)
        prior = x[:, 2:5]                                    # (B, 3, D, H, W), zyx components
        out = torch.zeros(x.shape[0], 6, *x.shape[2:], dtype=x.dtype)
        for b in range(x.shape[0]):
            p = prior[b][:, cond[b]].mean(dim=1)
            p = p / (p.norm() + 1e-8)
            out[b, 0:3][:, cond[b]] = (4.0 * p)[:, None]      # front neighbour 4 voxels along +prior
            out[b, 3:6][:, cond[b]] = (-4.0 * p)[:, None]     # back neighbour 4 voxels along -prior
        return {"displacement": out}


def _inputs():
    x = torch.zeros(1, 8, 16, 16, 16)
    x[0, 1, 8, 4:12, 4:12] = 1.0           # conditioning sheet at z = 8
    x[0, 2, 8, 4:12, 4:12] = 1.0           # +prior = +z
    x[0, 5, 8, 4:12, 4:12] = -1.0          # -prior = -z
    return x


def _run(x, **kwargs):
    model = _SlotContractModel()
    return run_model_tta(
        model, x, False, None,
        get_displacement_result=lambda m, inp, a, d: m(inp)["displacement"],
        merge_method="mean", outlier_drop_thresh=None, **kwargs,
    )


def _slot_a_at_cond(disp, x):
    cond = x[0, 1] > 0
    return disp[0, 0:3][:, cond].mean(dim=1)


def test_mirror_tta_without_prior_transform_collapses_the_merge():
    x = _inputs()
    merged = _run(x, transform_mode="mirror")
    # 4 of 8 variants flip z and see a prior pointing at the back neighbour: the mean of (+4) and (-4) is 0
    assert abs(float(_slot_a_at_cond(merged, x)[0])) < 1e-4


def test_mirror_tta_with_prior_transform_matches_identity():
    x = _inputs()
    merged = _run(x, transform_mode="mirror", input_vector_channels=(2, 5))
    assert torch.allclose(_slot_a_at_cond(merged, x), torch.tensor([4.0, 0.0, 0.0]), atol=1e-4)


def test_rotate_tta_with_prior_transform_matches_identity():
    x = _inputs()
    merged = _run(x, transform_mode="rotate3", input_vector_channels=(2, 5))
    assert torch.allclose(_slot_a_at_cond(merged, x), torch.tensor([4.0, 0.0, 0.0]), atol=1e-4)
