import pytest
import numpy

from chaospy import E, Iid, Uniform, Normal
from chaospy.quadrature import quad_gaussian, RECURRENCE_ALGORITHMS


@pytest.fixture(params=RECURRENCE_ALGORITHMS)
def recurrence_algorithm(request):
    yield request.param


def test_1d_gauss_hermite_quadrature(recurrence_algorithm):
    distribution = Normal(2, 2)
    abscissas, weights = quad_gaussian(
        10, distribution, recurrence_algorithm=recurrence_algorithm)
    assert abscissas.shape == (1, 11)
    assert weights.shape == (11,)
    assert numpy.allclose(numpy.sum(abscissas*weights, -1), distribution.mom(1))
    assert numpy.allclose(numpy.sum(abscissas**2*weights, -1), distribution.mom(2))


def test_3d_gauss_hermite_quadrature(recurrence_algorithm):
    distribution = Iid(Normal(0, 1), 3)
    abscissas, weights = quad_gaussian(
        3, distribution, recurrence_algorithm=recurrence_algorithm)
    assert abscissas.shape == (3, 4**3)
    assert weights.shape == (4**3,)
    kloc = numpy.eye(3, dtype=int)
    assert numpy.allclose(numpy.sum(abscissas*weights, -1),
                          distribution.mom(kloc))
    assert numpy.allclose(numpy.sum(abscissas**2*weights, -1),
                          distribution.mom(2*kloc))
