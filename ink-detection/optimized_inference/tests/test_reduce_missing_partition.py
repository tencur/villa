import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import processing  # noqa: E402


def write_array(directory, name, part_id, data):
    """Write one partition array the way inference.predict_fn does."""
    array = zarr.open(
        os.path.join(directory, f"{name}_part_{part_id:03d}.zarr"),
        mode="w",
        shape=data.shape,
        chunks=(1024, 1024),
        dtype=np.float32,
        compressor=None,
        zarr_format=2,
    )
    array[:] = data


def write_partition(directory, part_id, rows, shape=(8, 4)):
    """A partition that predicts 1.0 on `rows` and holds nothing elsewhere."""
    count = np.zeros(shape, dtype=np.float32)
    count[rows] = 0.5
    write_array(directory, "mask_pred", part_id, count)
    write_array(directory, "mask_count", part_id, count)


class ReduceMissingPartitionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.parts = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_reduce_refuses_to_blend_when_a_partition_is_missing(self):
        # Four partitions of two image rows each; the pod for partition 2 never finished.
        for part_id in (0, 1, 3):
            write_partition(self.parts, part_id, slice(2 * part_id, 2 * part_id + 2))

        with self.assertRaisesRegex(RuntimeError, r"1 of 4 partitions are missing.*\[2\]"):
            processing.reduce_partitions(self.parts, 4, (8, 4), tile_size=1024)

    def test_reduce_refuses_a_partition_without_its_count_array(self):
        write_partition(self.parts, 0, slice(0, 4))
        write_partition(self.parts, 1, slice(4, 8))
        os.rename(
            os.path.join(self.parts, "mask_count_part_001.zarr"),
            os.path.join(self.parts, "mask_count_part_001.zarr.incomplete"),
        )

        with self.assertRaisesRegex(RuntimeError, r"1 of 2 partitions are missing.*\[1\]"):
            processing.reduce_partitions(self.parts, 2, (8, 4), tile_size=1024)


if __name__ == "__main__":
    unittest.main()
