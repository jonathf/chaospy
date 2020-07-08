"""
Generate the quadrature abscissas and weights for simple grid.

Available to ensure that discrete distributions works along side
continuous ones.

Example usage
-------------

The first few orders with linear growth rule::

    >>> distribution = chaospy.DiscreteUniform(-2, 2)
    >>> for order in [0, 1, 2, 3, 4, 5, 9]:
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="discrete")
    ...     print(order, abscissas.round(3), weights.round(3))
    0 [[0]] [1.]
    1 [[-1  1]] [0.5 0.5]
    2 [[-1  0  1]] [0.333 0.333 0.333]
    3 [[-1  0  0  1]] [0.25 0.25 0.25 0.25]
    4 [[-2  0  0  1  2]] [0.2 0.2 0.2 0.2 0.2]
    5 [[-2  0  0  1  2]] [0.2 0.2 0.2 0.2 0.2]
    9 [[-2  0  0  1  2]] [0.2 0.2 0.2 0.2 0.2]

As the accuracy of discrete distribution plateau when all contained values are included, there is no reason to increase the number of nodes after this point.
"""
import numpy

from .combine import combine_quadrature
from .grid import quad_grid


def quad_discrete(order, domain=(0, 1)):
    """
    Generate quadrature abscissas and weights for discrete distributions.

    Same as regular grid, but `order` plateau at the `upper-lower-1`.
    At this order, finite state discrete distributions are analytically
    correct, and higher order will make the accuracy worsen.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (chaospy.distributions.baseclass.Dist, numpy.ndarray):

    Returns:
        (numpy.ndarray, numpy.ndarray):
            The quadrature points and weights. The points are
            equi-spaced grid on the interior of the domain bounds.
            The weights are all equal to `1/len(weights[0])`.
            Either distribution or bounding of interval to integrate over.

    Examples:
        >>> distribution = chaospy.DiscreteUniform(-2, 2)
        >>> abscissas, weights = chaospy.quad_discrete(4, distribution)
        >>> abscissas.round(4)
        array([[-2., -1.,  0.,  1.,  2.]])
        >>> weights.round(4)
        array([0.2, 0.2, 0.2, 0.2, 0.2])
        >>> abscissas, weights = chaospy.quad_discrete(9, distribution)
        >>> abscissas.round(4)
        array([[-2., -1.,  0.,  1.,  2.]])
        >>> weights.round(4)
        array([0.2, 0.2, 0.2, 0.2, 0.2])

    """
    from ..distributions.baseclass import Dist
    if isinstance(domain, Dist):
        abscissas, weights = quad_discrete(order, (domain.lower, domain.upper))
        weights *= domain.pdf(abscissas).flatten()
        weights /= numpy.sum(weights)
        return abscissas, weights

    order = numpy.atleast_1d(order)
    order, lower, upper = numpy.broadcast_arrays(order, domain[0], domain[1])
    assert order.ndim == 1, "too many dimensions"

    order_max = numpy.round(upper-lower).astype(int)-1
    order = numpy.where(order > order_max, order_max, order)

    return quad_grid(order, (lower, upper))
