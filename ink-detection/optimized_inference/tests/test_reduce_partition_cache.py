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


def write_partition(directory, part_id, pred, count):
    """Write one partition the way inference.predict_fn does."""
    os.makedirs(directory, exist_ok=True)
    for name, data in (("mask_pred", pred), ("mask_count", count)):
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


def write_run(directory, shape, probability, num_parts=1):
    """A run whose blended prediction is `probability` everywhere."""
    for part_id in range(num_parts):
        count = np.full(shape, 0.5, dtype=np.float32)
        write_partition(directory, part_id, count * probability, count)


def reduce_image(directory, shape, num_parts=1):
    tiles, _ = processing.reduce_partitions(directory, num_parts, shape, tile_size=1024)
    tiles = list(tiles)
    assert len(tiles) == 1  # every shape used here fits in one output tile
    return tiles[0]


class ReducePartitionCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        # Keep the reduce cache inside the test directory.
        self._saved_tempdir = tempfile.tempdir
        tempfile.tempdir = os.path.join(self.tmp, "tmp")
        os.makedirs(tempfile.tempdir)

    def tearDown(self):
        tempfile.tempdir = self._saved_tempdir
        self._tmp.cleanup()

    def run_dir(self, name):
        return os.path.join(self.tmp, name)

    def test_second_run_is_not_blended_from_first_run(self):
        shape = (8, 12)
        write_run(self.run_dir("a"), shape, 0.25)
        write_run(self.run_dir("b"), shape, 0.75)

        first = reduce_image(self.run_dir("a"), shape)
        second = reduce_image(self.run_dir("b"), shape)

        np.testing.assert_array_equal(first, np.full(shape, 63, dtype=np.uint8))
        np.testing.assert_array_equal(second, np.full(shape, 191, dtype=np.uint8))

    def test_smaller_second_run_is_not_a_crop_of_first_run(self):
        write_run(self.run_dir("a"), (8, 12), 0.25)
        write_run(self.run_dir("b"), (4, 6), 0.75)

        reduce_image(self.run_dir("a"), (8, 12))
        second = reduce_image(self.run_dir("b"), (4, 6))

        np.testing.assert_array_equal(second, np.full((4, 6), 191, dtype=np.uint8))

    def test_larger_second_run_does_not_fail_on_first_run_shape(self):
        write_run(self.run_dir("a"), (4, 6), 0.25)
        write_run(self.run_dir("b"), (8, 12), 0.75)

        reduce_image(self.run_dir("a"), (4, 6))
        second = reduce_image(self.run_dir("b"), (8, 12))

        np.testing.assert_array_equal(second, np.full((8, 12), 191, dtype=np.uint8))

    def test_rerun_with_fewer_partitions_ignores_earlier_partitions(self):
        shape = (8, 12)
        write_run(self.run_dir("a"), shape, 0.25, num_parts=3)
        write_run(self.run_dir("b"), shape, 0.75, num_parts=2)

        reduce_image(self.run_dir("a"), shape, num_parts=3)
        second = reduce_image(self.run_dir("b"), shape, num_parts=2)

        np.testing.assert_array_equal(second, np.full(shape, 191, dtype=np.uint8))

    def test_partitions_of_one_run_are_blended_together(self):
        shape = (8, 12)
        ones = np.ones(shape, dtype=np.float32)
        write_partition(self.run_dir("a"), 0, ones * 0.25, ones)
        write_partition(self.run_dir("a"), 1, ones * 0.75, ones)

        image = reduce_image(self.run_dir("a"), shape, num_parts=2)

        np.testing.assert_array_equal(image, np.full(shape, 127, dtype=np.uint8))

    def test_cache_is_removed_once_tiles_are_produced(self):
        shape = (8, 12)
        write_run(self.run_dir("a"), shape, 0.25)

        reduce_image(self.run_dir("a"), shape)

        self.assertEqual(os.listdir(tempfile.tempdir), [])


if __name__ == "__main__":
    unittest.main()
