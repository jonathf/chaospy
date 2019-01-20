"""
Fejér quadrature method.

Example usage
-------------

The first few orders with linear growth rule::

    >>> for order in [0, 1, 2, 3]:
    ...     abscissas, weights = quad_fejer(order)
    ...     print(order, numpy.around(abscissas, 3), numpy.around(weights, 3))
    0 [[0.5]] [1.]
    1 [[0.25 0.75]] [0.444 0.444]
    2 [[0.146 0.5   0.854]] [0.267 0.4   0.267]
    3 [[0.095 0.345 0.655 0.905]] [0.18 0.3  0.3  0.18]

The first few orders with exponential growth rule::

    >>> for order in [0, 1, 2]:
    ...     abscissas, weights = quad_fejer(order, growth=True)
    ...     print(order, numpy.around(abscissas, 3), numpy.around(weights, 3))
    0 [[0.5]] [1.]
    1 [[0.146 0.5   0.854]] [0.267 0.4   0.267]
    2 [[0.038 0.146 0.309 0.5   0.691 0.854 0.962]] [0.073 0.14  0.181 0.197 0.181 0.14  0.073]
"""
from __future__ import division

import numpy
import chaospy.quad


def quad_fejer(order, lower=0, upper=1, growth=False, part=None):
    """
    Generate the quadrature abscisas and weights in Fejér quadrature.

    Example:
        >>> abscissas, weights = quad_fejer(3, 0, 1)
        >>> print(numpy.around(abscissas, 4))
        [[0.0955 0.3455 0.6545 0.9045]]
        >>> print(numpy.around(weights, 4))
        [0.1804 0.2996 0.2996 0.1804]
    """
    order = numpy.asarray(order, dtype=int).flatten()
    lower = numpy.asarray(lower).flatten()
    upper = numpy.asarray(upper).flatten()

    dim = max(lower.size, upper.size, order.size)

    order = numpy.ones(dim, dtype=int)*order
    lower = numpy.ones(dim)*lower
    upper = numpy.ones(dim)*upper

    composite = numpy.array([numpy.arange(2)]*dim)

    if growth:
        results = [
            _fejer(numpy.where(order[i] == 0, 0, 2.**(order[i]+1)-2))
            for i in range(dim)
        ]
    else:
        results = [
            _fejer(order[i], composite[i]) for i in range(dim)
        ]

    abscis = [_[0] for _ in results]
    weight = [_[1] for _ in results]

    abscis = chaospy.quad.combine(abscis, part=part).T
    weight = chaospy.quad.combine(weight, part=part)

    abscis = ((upper-lower)*abscis.T + lower).T
    weight = numpy.prod(weight*(upper-lower), -1)

    assert len(abscis) == dim
    assert len(weight) == len(abscis.T)

    return abscis, weight


def _fejer(order, composite=None):
    r"""
    Backend method.

    Examples:
        >>> print(*_fejer(0))
        [0.5] [1.]
        >>> print(*_fejer(1))
        [0.25 0.75] [0.44444444 0.44444444]
        >>> print(*_fejer(2))
        [0.14644661 0.5        0.85355339] [0.26666667 0.4        0.26666667]
        >>> print(*_fejer(3), sep="\n")
        [0.0954915 0.3454915 0.6545085 0.9045085]
        [0.18037152 0.29962848 0.29962848 0.18037152]
        >>> print(*_fejer(4), sep="\n")
        [0.0669873 0.25      0.5       0.75      0.9330127]
        [0.12698413 0.22857143 0.26031746 0.22857143 0.12698413]
        >>> print(*_fejer(5), sep="\n")
        [0.04951557 0.1882551  0.38873953 0.61126047 0.8117449  0.95048443]
        [0.0950705  0.17612121 0.2186042  0.2186042  0.17612121 0.0950705 ]
    """
    order = int(order)
    if order == 0:
        return numpy.array([.5]), numpy.array([1.])

    order += 2

    theta = (order-numpy.arange(order+1))*numpy.pi/order
    abscisas = 0.5*numpy.cos(theta) + 0.5

    N, K = numpy.mgrid[:order+1, :order//2]
    weights = 2*numpy.cos(2*(K+1)*theta[N])/(4*K*(K+2)+3)
    if order % 2 == 0:
        weights[:, -1] *= 0.5
    weights = (1-numpy.sum(weights, -1)) / order

    return abscisas[1:-1], weights[1:-1]
