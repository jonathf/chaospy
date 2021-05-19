"""
Check the creation of quadrature nodes.

Create Gaussian quadrature nodes using various distributions and algorithms and
check if the nodes correctly can be used to estimate raw statistical nodes up
to 2N-1. Check for both 1 and 3 dimensions.
"""
import pytest
import numpy

import chaospy


def test_1d_quadrature_creation(
        analytical_distribution, recurrence_algorithm):
    """Check 1-D quadrature rule."""
    abscissas, weights = chaospy.quadrature.gaussian(
        order=8,
        dist=analytical_distribution,
        recurrence_algorithm=recurrence_algorithm,
    )
    assert abscissas.shape == (1, 9)
    assert weights.shape == (9,)
    assert numpy.allclose(numpy.sum(abscissas*weights, -1),
                          analytical_distribution.mom(1))
    assert numpy.allclose(numpy.sum(abscissas**2*weights, -1),
                          analytical_distribution.mom(2))
    # lanczos not working as well as the others for heavy tails:
    rtol = 1e-3 if recurrence_algorithm == "lanczos" else 1e-5
    assert numpy.allclose(numpy.sum(abscissas**15*weights, -1),
                          analytical_distribution.mom(15), rtol=rtol)


def test_3d_quadrature_creation(
        analytical_distribution, recurrence_algorithm):
    """Check 3-D quadrature rule."""
    distribution = chaospy.Iid(analytical_distribution, 3)
    abscissas, weights = chaospy.quadrature.gaussian(
        order=3,
        dist=distribution,
        recurrence_algorithm=recurrence_algorithm,
    )
    assert abscissas.shape == (3, 4**3)
    assert weights.shape == (4**3,)
    kloc = numpy.eye(3, dtype=int)
    assert numpy.allclose(numpy.sum(abscissas*weights, -1),
                          distribution.mom(kloc))
    assert numpy.allclose(numpy.sum(abscissas**2*weights, -1),
                          distribution.mom(2*kloc))
    assert numpy.allclose(numpy.sum(abscissas**5*weights, -1),
                          distribution.mom(5*kloc))
