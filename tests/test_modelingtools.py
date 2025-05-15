import numpy as np
from pybigue.utils import angle_modulo
import pytest


def test_anglemodulo_loopsNegative():
    assert pytest.approx(angle_modulo(.9*np.pi + .2*np.pi,), abs=1e-5) == -.9*np.pi


def test_anglemodulo_loopsPositive():
    assert pytest.approx(angle_modulo(-.9*np.pi - .2*np.pi,), abs=1e-5) == .9*np.pi


def test_anglemodulo_largeNegativeAngle():
    assert pytest.approx(angle_modulo(0.6 - 2000*np.pi,), abs=1e-5) == 0.6


def test_anglemodulo_largePositiveAngle():
    assert pytest.approx(angle_modulo(0.6 + 2000*np.pi,), abs=1e-5) == 0.6
