r"""
Clenshaw-Curtis quadrature method.

Example usage
-------------

The first few orders with linear growth rule::

    >>> for order in [0, 1, 2, 3]:
    ...     abscissas, weights = quad_clenshaw_curtis(order)
    ...     print(order, numpy.around(abscissas, 3), numpy.around(weights, 3))
    0 [[0.5]] [1.]
    1 [[0. 1.]] [0.5 0.5]
    2 [[0.  0.5 1. ]] [0.167 0.667 0.167]
    3 [[0.   0.25 0.75 1.  ]] [0.056 0.444 0.444 0.056]

The first few orders with exponential growth rule::

    >>> for order in [0, 1, 2]:
    ...     abscissas, weights = quad_clenshaw_curtis(order, growth=True)
    ...     print(order, numpy.around(abscissas, 3), numpy.around(weights, 3))
    0 [[0.5]] [1.]
    1 [[0.  0.5 1. ]] [0.167 0.667 0.167]
    2 [[0.    0.146 0.5   0.854 1.   ]] [0.033 0.267 0.4   0.267 0.033]
"""
from __future__ import division

import numpy
import chaospy.quad


def quad_clenshaw_curtis(order, lower=0, upper=1, growth=False, part=None):
    """
    Generate the quadrature nodes and weights in Clenshaw-Curtis quadrature.

    Example:
        >>> abscissas, weights = quad_clenshaw_curtis(3, 0, 1)
        >>> print(numpy.around(abscissas, 4))
        [[0.   0.25 0.75 1.  ]]
        >>> print(numpy.around(weights, 4))
        [0.0556 0.4444 0.4444 0.0556]
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
            _clenshaw_curtis(2**order[i]-1*(order[i] == 0), composite[i])
            for i in range(dim)
        ]
    else:
        results = [
            _clenshaw_curtis(order[i], composite[i]) for i in range(dim)
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


def _clenshaw_curtis(order, composite=None):
    r"""
    Backend method.

    Examples:
        >>> print(*_clenshaw_curtis(0))
        [0.5] [1.]
        >>> print(*_clenshaw_curtis(1))
        [0. 1.] [0.5 0.5]
        >>> print(*_clenshaw_curtis(2))
        [0.  0.5 1. ] [0.16666667 0.66666667 0.16666667]
        >>> print(*_clenshaw_curtis(3))
        [0.   0.25 0.75 1.  ] [0.05555556 0.44444444 0.44444444 0.05555556]
        >>> print(*_clenshaw_curtis(4), sep="\n")
        [0.         0.14644661 0.5        0.85355339 1.        ]
        [0.03333333 0.26666667 0.4        0.26666667 0.03333333]
        >>> print(*_clenshaw_curtis(5), sep="\n")
        [0.        0.0954915 0.3454915 0.6545085 0.9045085 1.       ]
        [0.02       0.18037152 0.29962848 0.29962848 0.18037152 0.02      ]
    """
    if order == 0:
        return numpy.array([.5]), numpy.array([1.])

    theta = (order-numpy.arange(order+1))*numpy.pi/order
    abscisas = 0.5*numpy.cos(theta) + 0.5

    N, K = numpy.mgrid[:order+1, :order//2]
    weights = 2*numpy.cos(2*(K+1)*theta[N])/(4*K*(K+2)+3)
    if order % 2 == 0:
        weights[:, -1] *= 0.5
    weights = (1-numpy.sum(weights, -1)) / order

    weights[0] /= 2
    weights[-1] /= 2

    return abscisas, weights
