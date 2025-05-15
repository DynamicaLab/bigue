import pytest
import numpy as np

from pybigue.kernels.clusters import clockwise_separation


def iter_approx(values):
    return type(values)(map(pytest.approx, values))

def test_clockwise_separation():
    assert pytest.approx(clockwise_separation(0., -2)) == 2
    assert pytest.approx(clockwise_separation(3., -2)) == 5
    assert pytest.approx(clockwise_separation(2, 3)) == 2*np.pi-1

    assert iter_approx(clockwise_separation(
        np.array([0., 3, 2.]), np.array([-2., -2., 3.])).tolist()) == [2, 5, 2*np.pi-1]
