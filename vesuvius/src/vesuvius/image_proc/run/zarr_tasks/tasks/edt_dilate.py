"""EDT-based mask dilation task for zarr arrays.

Creates a dilated binary mask from a zarr using Euclidean Distance Transform.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
import zarr

from vesuvius.data.utils import create_zarr_array, open_zarr_group
from numcodecs import Blosc

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task

# Try to import edt package for fast EDT
try:
    import edt as edt_package

    HAS_EDT = True
except ImportError:
    HAS_EDT = False
    from scipy.ndimage import distance_transform_edt as scipy_edt


# Global state for worker processes
_WORKER_STATE = {}


def _init_worker(input_path: str, output_path: str, resolution: str) -> None:
    """Initialize worker with zarr paths."""
    global _WORKER_STATE
    _WORKER_STATE = {
        "input_path": input_path,
        "output_path": output_path,
        "resolution": resolution,
    }


@dataclass
class EdtDilateConfig(TaskConfig):
    """Configuration for EDT dilation task."""

    distance: float = 10.0
    chunk_size: Tuple[int, int, int] = (128, 256, 256)
    black_border: bool = True
    resolution: str = "0"


def _edt_dilate_worker(
    args: Tuple,
) -> Tuple[Tuple[int, int, int], bool]:
    """Process a single chunk: create mask, compute EDT, threshold.

    Returns:
        (chunk_idx, was_written)
    """
    global _WORKER_STATE
    chunk_idx, chunk_bounds, distance, black_border = args

    input_path = _WORKER_STATE["input_path"]
    output_path = _WORKER_STATE["output_path"]
    resolution = _WORKER_STATE["resolution"]

    z_start, z_end, y_start, y_end, x_start, x_end = chunk_bounds

    # Read the chunk with a halo wider than the dilation distance, so that foreground
    # just across a chunk face dilates into this chunk. Without it every chunk was
    # dilated on its own and the dilation stopped at each seam.
    input_arr = zarr.open(input_path, mode="r")[resolution]
    halo = int(np.ceil(distance)) + 1
    shape = input_arr.shape
    lo = [max(0, start - halo) for start in (z_start, y_start, x_start)]
    hi = [min(size, end + halo) for size, end in zip(shape, (z_end, y_end, x_end))]
    block = input_arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # Create binary mask of nonzero values
    mask = block != 0

    # Skip if no foreground is within reach of this chunk
    if not mask.any():
        return (chunk_idx, False)

    # Invert mask for EDT (EDT computes distance from True voxels)
    inverted = ~mask

    # Compute EDT. In the inverted mask foreground is 0, so the edt package's
    # black_border=True would treat everything outside the block as foreground and
    # paint a shell along every face. --black-border (default True) means the volume
    # boundary is background, i.e. black_border=False here; the halo covers the seams.
    if HAS_EDT:
        # Ensure contiguity for edt package
        if not inverted.flags["C_CONTIGUOUS"]:
            inverted = np.ascontiguousarray(inverted)
        # Use single thread since we parallelize over chunks
        distances = edt_package.edt(inverted, parallel=1, black_border=not black_border)
    else:
        distances = scipy_edt(inverted)
    distances = distances[
        z_start - lo[0]:z_end - lo[0],
        y_start - lo[1]:y_end - lo[1],
        x_start - lo[2]:x_end - lo[2],
    ]

    # Threshold to create dilated mask
    dilated = (distances <= distance).astype(np.uint8)

    # Write to output
    output_arr = zarr.open(output_path, mode="r+")[resolution]
    output_arr[z_start:z_end, y_start:y_end, x_start:x_end] = dilated

    return (chunk_idx, True)


@register_task("edt-dilate")
class EdtDilateTask(ZarrTask):
    """Create a dilated binary mask using Euclidean Distance Transform."""

    use_executor = True  # Use ProcessPoolExecutor for initializer support

    def __init__(self, config: EdtDilateConfig):
        super().__init__(config)
        self.config: EdtDilateConfig = config
        self._shape: Tuple[int, ...] = ()
        self._chunks_written: int = 0

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add EDT-dilate-specific arguments."""
        parser.add_argument(
            "--distance",
            type=float,
            required=True,
            help="EDT threshold distance for dilation in voxels",
        )
        parser.add_argument(
            "--chunk-size",
            type=str,
            default="128,256,256",
            help="Processing chunk size z,y,x (default: 128,256,256)",
        )
        parser.add_argument(
            "--black-border",
            type=lambda x: x.lower() in ("true", "1", "yes"),
            default=True,
            help="Treat space outside the volume as background (default: True); false treats it as foreground",
        )
        parser.add_argument(
            "--resolution",
            type=str,
            default="0",
            help="Resolution level to process (default: '0')",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EdtDilateTask":
        """Create task from parsed arguments."""
        # Parse chunk size
        chunk_size = tuple(int(x) for x in args.chunk_size.split(","))
        if len(chunk_size) != 3:
            raise ValueError("chunk-size must have exactly 3 values (z,y,x)")

        base_config = make_task_config(args)
        config = EdtDilateConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=base_config.output_zarr,
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            distance=args.distance,
            chunk_size=chunk_size,
            black_border=args.black_border,
            resolution=getattr(args, "resolution", "0"),
        )
        return cls(config)

    def get_worker_initializer(self) -> Optional[Tuple[Callable, Tuple]]:
        """Get initializer for ProcessPoolExecutor workers."""
        return (
            _init_worker,
            (
                self.config.input_zarr,
                self.config.output_zarr,
                self.config.resolution,
            ),
        )

    def prepare(self) -> None:
        """Create output zarr structure."""
        if HAS_EDT:
            print("Using edt package for EDT")
        else:
            print(
                "Using scipy (slower) for EDT - install 'edt' package for faster processing"
            )

        # Open input zarr
        input_store = zarr.open(self.config.input_zarr, mode="r")
        input_arr = input_store[self.config.resolution]
        self._shape = input_arr.shape

        print(f"Input shape: {self._shape}, dtype: {input_arr.dtype}")
        print(f"Resolution level: {self.config.resolution}")
        print(f"Dilation distance: {self.config.distance} voxels")
        print(f"Processing chunk size: {self.config.chunk_size}")
        print(f"Black border: {self.config.black_border}")

        # Create output zarr
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

        output_store = open_zarr_group(self.config.output_zarr, mode="w")
        output_arr = create_zarr_array(
            output_store,
            self.config.resolution,
            shape=self._shape,
            chunks=self.config.chunk_size,
            dtype=np.uint8,
            compressor=compressor,
            fill_value=0,
            write_empty_chunks=False,
        )

        print(f"Created output zarr: {self.config.output_zarr}")
        print(
            f"  shape={output_arr.shape}, chunks={output_arr.chunks}, dtype={output_arr.dtype}"
        )

    def generate_work_items(self) -> Iterable[Any]:
        """Generate chunk coordinates for parallel processing."""
        cz_size, cy_size, cx_size = self.config.chunk_size

        n_chunks_z = (self._shape[0] + cz_size - 1) // cz_size
        n_chunks_y = (self._shape[1] + cy_size - 1) // cy_size
        n_chunks_x = (self._shape[2] + cx_size - 1) // cx_size

        for cz in range(n_chunks_z):
            for cy in range(n_chunks_y):
                for cx in range(n_chunks_x):
                    z_start = cz * cz_size
                    y_start = cy * cy_size
                    x_start = cx * cx_size
                    z_end = min(z_start + cz_size, self._shape[0])
                    y_end = min(y_start + cy_size, self._shape[1])
                    x_end = min(x_start + cx_size, self._shape[2])

                    chunk_idx = (cz, cy, cx)
                    chunk_bounds = (z_start, z_end, y_start, y_end, x_start, x_end)
                    yield (
                        chunk_idx,
                        chunk_bounds,
                        self.config.distance,
                        self.config.black_border,
                    )

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _edt_dilate_worker(args)

    def finalize(self) -> None:
        """Print completion summary."""
        results = getattr(self, "_results", [])
        chunks_written = sum(1 for _, written in results if written)
        total_chunks = len(results)

        print(f"\nComplete!")
        print(f"  Chunks written: {chunks_written}/{total_chunks}")
        print(f"  Output: {self.config.output_zarr}")
