"""
Generate the quadrature abscissas and weights for simple grid.

Mostly available to ensure that discrete distributions works along side
continuous ones.
"""
import numpy
import chaospy

from .combine import combine_quadrature


def quad_grid(order, domain=(0, 1)):
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
        >>> abscissas, weights = chaospy.quad_grid(4, chaospy.Uniform(-1, 1))
        >>> abscissas.round(4)
        array([[-0.8, -0.4,  0. ,  0.4,  0.8]])
        >>> weights.round(4)
        array([0.2, 0.2, 0.2, 0.2, 0.2])
        >>> abscissas, weights = chaospy.quad_grid([1, 1])
        >>> abscissas.round(4)
        array([[0.25, 0.25, 0.75, 0.75],
               [0.25, 0.75, 0.25, 0.75]])
        >>> weights.round(4)
        array([0.25, 0.25, 0.25, 0.25])

    """
    if isinstance(domain, chaospy.Distribution):
        abscissas, weights = quad_grid(order, (domain.lower, domain.upper))
        weights *= domain.pdf(abscissas).flatten()
        weights /= numpy.sum(weights)
        return abscissas, weights

    order = numpy.atleast_1d(order)
    order, lower, upper = numpy.broadcast_arrays(order, domain[0], domain[1])
    assert order.ndim == 1, "too many dimensions"
    abscissas = tuple(numpy.linspace(0, 1, 2*order_+3)[1::2] for order_ in order)
    weights = tuple(numpy.repeat(1./(order_+1), order_+1) for order_ in order)

    abscissas_, weights_ = combine_quadrature(abscissas, weights, (lower, upper))
    return abscissas_, weights_
