"""vesuvius.predict must normalize a vesuvius.train 'zscore' checkpoint the way training did.

vesuvius.train's zscore normalizer uses each patch's own mean and std. A checkpoint can still carry
dataset intensity properties (--no-skip-intensity-sampling, or intensity_properties in the config);
predict then used them for a global z-score, so the network saw inputs it was never trained on.
"""

import numpy as np
import torch
import zarr

from vesuvius.data.volume import Volume
from vesuvius.models.run import inference
from vesuvius.models.run.inference import Inferer
from vesuvius.models.training.normalization import get_normalization

PROPS = {'mean': 14.37, 'std': 30.19, 'percentile_00_5': 0.0, 'percentile_99_5': 160.0}


def _inferer(props):
    inferer = Inferer.__new__(Inferer)
    inferer.model_normalization_scheme = 'zscore'
    inferer.model_intensity_properties = props
    inferer.normalization_scheme = 'instance_zscore'
    for name, value in dict(input='unused.zarr', patch_size=(8, 8, 8), overlap=0.5, num_parts=1, part_id=0,
                            input_format='zarr', verbose=False, skip_empty_patches=False, scroll_id=None,
                            segment_id=None, energy=None, resolution=None, input_anon=False, bbox=None,
                            read_retries=1, device=torch.device('cpu'), max_patches=None).items():
        setattr(inferer, name, value)
    return inferer


def _captured(monkeypatch, props):
    captured = {}

    class FakeDataset:
        collate_fn = staticmethod(lambda batch: batch)

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.all_positions = []

        def __len__(self):
            return 0

    monkeypatch.setattr(inference, 'VCDataset', FakeDataset)
    _inferer(props)._create_dataset_and_loader()
    return captured


def test_zscore_checkpoint_with_intensity_properties_stays_per_patch(monkeypatch):
    captured = _captured(monkeypatch, dict(PROPS))
    assert captured['normalization_scheme'] == 'instance_zscore'
    assert captured['global_mean'] is None and captured['global_std'] is None


def test_zscore_checkpoint_without_properties_unchanged(monkeypatch):
    captured = _captured(monkeypatch, None)
    assert captured['normalization_scheme'] == 'instance_zscore'


def test_predict_input_matches_training_input(tmp_path):
    rng = np.random.default_rng(0)
    patch = (rng.normal(60.0, 9.0, size=(8, 8, 8))).clip(0, 255).astype(np.uint8)
    arr = zarr.open(str(tmp_path / 'in.zarr'), mode='w', shape=patch.shape, chunks=patch.shape, dtype=patch.dtype)
    arr[:] = patch
    training_input = get_normalization('zscore', PROPS).run(patch.astype(np.float32))
    volume = Volume(type='zarr', path=str(tmp_path / 'in.zarr'),
                    normalization_scheme=inference.map_train_zscore_scheme('zscore'),
                    return_as_type='np.float32', return_as_tensor=False)
    np.testing.assert_allclose(volume[0:8, 0:8, 0:8], training_input, atol=1e-4)
