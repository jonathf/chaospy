"""
Generate the quadrature abscissas and weights for simple grid.

Mostly available to ensure that discrete distributions works along side
continuous ones.
"""
import numpy
import chaospy

from .hypercube import hypercube_quadrature


def grid(order, domain=(0, 1), growth=False, segments=1):
    """
    Generate the quadrature abscissas and weights for simple grid.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (chaospy.distributions.baseclass.Distribution, numpy.ndarray):
            Either distribution or bounding of interval to integrate over.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            The quadrature points and weights. The points are
            equi-spaced grid on the interior of the domain bounds.
            The weights are all equal to `1/len(weights[0])`.

    Example:
        >>> abscissas, weights = chaospy.quadrature.grid(4, chaospy.Uniform(-1, 1))
        >>> abscissas.round(4)
        array([[-0.8, -0.4,  0. ,  0.4,  0.8]])
        >>> weights.round(4)
        array([0.2, 0.2, 0.2, 0.2, 0.2])
        >>> abscissas, weights = chaospy.quadrature.grid([1, 1])
        >>> abscissas.round(4)
        array([[0.25, 0.25, 0.75, 0.75],
               [0.25, 0.75, 0.25, 0.75]])
        >>> weights.round(4)
        array([0.25, 0.25, 0.25, 0.25])

    """
    order = numpy.asarray(order)
    order = numpy.where(growth, numpy.where(order > 0, 3**order-1, 0), order)
    return hypercube_quadrature(
        quad_func=grid_simple,
        order=order,
        domain=domain,
        segments=segments,
    )


def grid_simple(order):
    """
    Backend for grid quadrature.

    Use :func:`chaospy.quadrature.grid` instead.
    """
    order = int(order)
    abscissas = numpy.linspace(0, 1, 2*order+3)[1::2]
    weights = numpy.full(order+1, 1./(order+1))
    return abscissas, weights
