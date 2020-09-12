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
    2 [[-2  0  2]] [0.333 0.333 0.333]
    3 [[-2 -1  1  2]] [0.25 0.25 0.25 0.25]
    4 [[-2 -1  0  1  2]] [0.2 0.2 0.2 0.2 0.2]
    5 [[-2 -1  0  1  2]] [0.2 0.2 0.2 0.2 0.2]
    9 [[-2 -1  0  1  2]] [0.2 0.2 0.2 0.2 0.2]

As the accuracy of discrete distribution plateau when all contained values are
included, there is no reason to increase the number of nodes after this point.

The first few orders with exponential growth rule where the nodes are nested::

    >>> distribution = chaospy.DiscreteUniform(0, 10)
    >>> for order in [0, 1, 2, 3, 4]:
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="discrete", growth=True)
    ...     print(order, abscissas)
    0 [[5]]
    1 [[1 5 9]]
    2 [[1 3 5 7 9]]
    3 [[ 0  1  3  4  5  6  7  9 10]]
    4 [[ 0  1  2  3  4  5  6  7  8  9 10]]
"""
import numpy
import chaospy

from .combine import combine_quadrature
from .grid import quad_grid


def quad_discrete(order, domain=(0, 1), growth=False, segments=1):
    """
    Generate quadrature abscissas and weights for discrete distributions.

    Same as regular grid, but `order` plateau at the `upper-lower-1`.
    At this order, finite state discrete distributions are analytically
    correct, and higher order will make the accuracy worsen.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (chaospy.distributions.baseclass.Distribution, numpy.ndarray):
            Either distribution or bounding of interval to integrate over.
        growth (bool):
            if true sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples.

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
    if isinstance(domain, chaospy.Distribution):
        abscissas, weights = quad_discrete(order, (domain.lower, domain.upper), growth=growth, segments=segments)
        weights *= domain.pdf(abscissas).flatten()
        weights /= numpy.sum(weights)
        return abscissas, weights

    del segments  # Basically segments doesn't do much here
    if growth:
        order = numpy.where(order > 0, 2**(order), 0)

    order = numpy.atleast_1d(order)
    order, lower, upper = numpy.broadcast_arrays(order, domain[0], domain[1])
    assert order.ndim == 1, "too many dimensions"

    order_max = numpy.round(upper-lower).astype(int)-1
    order = numpy.where(order > order_max, order_max, order)

    return quad_grid(order, (lower, upper))
