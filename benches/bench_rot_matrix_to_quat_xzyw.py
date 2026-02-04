import numpy as np
import numpy.typing as npt
import pytest
from pytest_benchmark.fixture import BenchmarkFixture
from scipy.spatial.transform import Rotation

from metaworld.utils.numpy import rotation_matrix_to_quat_xyzw


@pytest.fixture(scope="module")
def sample_matrix():
    """Generate a fixed random rotation matrix for consistent benchmarking."""
    rng = np.random.default_rng(42)
    return Rotation.random(random_state=rng).as_matrix()


def test_bench_custom_matrix_to_quat_xyzw(benchmark: BenchmarkFixture, sample_matrix: npt.NDArray):
    """Micro-benchmark for metaworld.utils.numpy.rotation_matrix_to_quat_xyzw."""
    # We benchmark the function call directly
    result = benchmark(rotation_matrix_to_quat_xyzw, sample_matrix)
    assert result.shape == (4,)


def test_bench_scipy_matrix_to_quat_xyzw(benchmark: BenchmarkFixture, sample_matrix: npt.NDArray):
    """Micro-benchmark for scipy.spatial.transform.Rotation.
    Includes the overhead of object creation (from_matrix) and extraction (as_quat).
    """

    def scipy_convert(R: npt.NDArray) -> npt.NDArray:
        return Rotation.from_matrix(R).as_quat()

    result = benchmark(scipy_convert, sample_matrix)
    assert result.shape == (4,)
