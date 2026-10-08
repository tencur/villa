"""The package must leave OpenCV able to decode images after it has been imported.

OpenCV reads OPENCV_IO_MAX_IMAGE_PIXELS when cv2 is imported and asserts ``pixels <= limit`` with no special case,
so the former default of "0" made every ``cv2.imread`` fail (observed with opencv-python-headless 4.11 through 5.0).
The test runs in a subprocess so the environment is exactly what the package sets.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cv2_can_decode_after_importing_the_package(tmp_path):
    script = textwrap.dedent(
        f"""
        import os, sys
        os.environ.pop("OPENCV_IO_MAX_IMAGE_PIXELS", None)
        sys.path.insert(0, {str(ROOT)!r})
        import processing  # sets the package default before cv2 is imported
        import cv2, numpy as np
        path = {str(tmp_path / "small.png")!r}
        cv2.imwrite(path, np.zeros((64, 64), np.uint8))
        assert cv2.imread(path, cv2.IMREAD_GRAYSCALE) is not None
        big = {str(tmp_path / "big.png")!r}
        cv2.imwrite(big, np.zeros((6000, 6000), np.uint8))   # a full-size layer is ~32 Mpx; stay well above that
        assert cv2.imread(big, cv2.IMREAD_GRAYSCALE).shape == (6000, 6000)
        print("ok")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout
