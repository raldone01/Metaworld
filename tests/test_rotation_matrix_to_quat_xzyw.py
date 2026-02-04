import numpy as np
import numpy.testing as nptest
from scipy.spatial.transform import Rotation

from metaworld.utils.numpy import rotation_matrix_to_quat_xyzw


def test_rotation_matrix_to_quat_xzyw():
    rng = np.random.default_rng(42)
    for _ in range(100):
        # Generate a random rotation matrix
        r = Rotation.random(random_state=rng)
        R = r.as_matrix()

        # Convert to quaternion using our function
        quat_xyzw = rotation_matrix_to_quat_xyzw(R)

        # Convert to quaternion using scipy
        quat_scipy = r.as_quat()  # Returns in (x, y, z, w) format

        if np.dot(quat_xyzw, quat_scipy) < 0:
            quat_xyzw = -quat_xyzw

        # Compare the two quaternions
        nptest.assert_allclose(quat_xyzw, quat_scipy, rtol=1e-6, atol=1e-6)
