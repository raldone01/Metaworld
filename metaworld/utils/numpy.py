from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit


def randint(rng: np.random.Generator, size: npt.NDArray | None = None) -> np.uint32 | np.ndarray:
    """Return a random integer from [0, 2**32 - 1] using the provided RNG."""
    return rng.integers(0, 2**32, size=size, dtype=np.uint32)


@njit
def rotation_matrix_to_quat_xyzw(R: npt.NDArray) -> npt.NDArray:
    """Convert a rotation matrix to a quaternion in (x, y, z, w) format.

    Args:
        R (npt.NDArray): A 3x3 rotation matrix.

    Returns:
        npt.NDArray: A quaternion in (x, y, z, w) format.

    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([x, y, z, w], dtype=R.dtype)
