"""Test segment functionality."""
import numpy
import chaospy

from chaospy.quadrature.hypercube import hypercube_quadrature, split_into_segments


def my_quad(order):
    return (numpy.linspace(0, 1, order+1)[numpy.newaxis], 1./numpy.full(order+1, order+2))


def test_segment_basic():
    """Ensure direct segment function works as expected."""
    abscissas, weights = my_quad(2)
    assert numpy.allclose(abscissas, [0., 1/2., 1.])
    assert numpy.allclose(weights, [1/4., 1/4., 1/4.])

    abscissas, weights = split_into_segments(my_quad, 4, segments=2)
    assert numpy.allclose(abscissas, [0., 0.25, 0.5, 0.75, 1.])
    assert numpy.allclose(weights, [1/8., 1/8., 2/8., 1/8., 1/8.])

    abscissas, weights = split_into_segments(my_quad, 4, segments=3)
    assert numpy.allclose(abscissas, [0., 1/6., 1/3., 4/6., 1.])
    assert numpy.allclose(weights, [3/36., 3/36., 7/36, 8/36., 4/36.])

    abscissas, weights = split_into_segments(my_quad, 4, segments=[0.2, 0.8])
    assert numpy.allclose(abscissas, [0. , 0.1, 0.2, 0.5, 0.8, 0.9, 1. ])
    assert numpy.allclose(weights, [0.05, 0.05, 0.2 , 0.15, 0.2 , 0.05, 0.05])


def test_hypercube_frontend():
    """Ensure indirect segment function works through hypercube_quadrature."""
    abscissas, weights = hypercube_quadrature(my_quad, (1, 1), domain=(0, 1))
    assert numpy.allclose(abscissas, [[0., 0., 1., 1.], [0., 1., 0., 1.]])
    assert numpy.allclose(weights, [1/9., 1/9., 1/9., 1/9.])

    abscissas, weights = hypercube_quadrature(my_quad, 2, domain=chaospy.Uniform(-1, 1))
    assert numpy.allclose(abscissas, [-1.,  0.,  1.])
    assert numpy.allclose(weights, [1/4., 1/4., 1/4.])
