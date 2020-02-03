"""
Clenshaw-Curtis quadrature method is a good all-around quadrature method
comparable to Gaussian quadrature, but typically limited to finite intervals
without a specific weight function. In addition to be quite accurate, the
weights and abscissas can be calculated quite fast.

Another thing to note is that Clenshaw-Curtis, with an appropriate growth rule
is fully nested. This means, if one applies a method that combines different
order of quadrature rules, the number of evaluations can often be reduced as
the abscissas can be used across levels.

Example usage
-------------

The first few orders with linear growth rule::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> for order in [0, 1, 2, 3]:
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="clenshaw_curtis")
    ...     print(order, abscissas.round(3), weights.round(3))
    0 [[0.5]] [1.]
    1 [[0. 1.]] [0.5 0.5]
    2 [[0.  0.5 1. ]] [0.167 0.667 0.167]
    3 [[0.   0.25 0.75 1.  ]] [0.056 0.444 0.444 0.056]

The first few orders with exponential growth rule::

    >>> for order in [0, 1, 2]:
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="clenshaw_curtis", growth=True)
    ...     print(order, abscissas.round(3), weights.round(3))
    0 [[0.5]] [1.]
    1 [[0.  0.5 1. ]] [0.167 0.667 0.167]
    2 [[0.    0.146 0.5   0.854 1.   ]] [0.033 0.267 0.4   0.267 0.033]

Applying the rule using Smolyak sparse grid::

    >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     2, distribution, rule="clenshaw_curtis",
    ...     growth=True, sparse=True)
    >>> abscissas.round(2)
    array([[0.  , 0.  , 0.  , 0.15, 0.5 , 0.5 , 0.5 , 0.5 , 0.5 , 0.85, 1.  ,
            1.  , 1.  ],
           [0.  , 0.5 , 1.  , 0.5 , 0.  , 0.15, 0.5 , 0.85, 1.  , 0.5 , 0.  ,
            0.5 , 1.  ]])
    >>> weights.round(3)
    array([ 0.028, -0.022,  0.028,  0.267, -0.022,  0.267, -0.089,  0.267,
           -0.022,  0.267,  0.028, -0.022,  0.028])
"""
from __future__ import division

import numpy

from .combine import combine_quadrature


def quad_clenshaw_curtis(order, domain, growth=False):
    """
    Generate the quadrature nodes and weights in Clenshaw-Curtis quadrature.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (chaospy.distributions.baseclass.Dist, numpy.ndarray):
            Either distribution or bounding of interval to integrate over.
        growth (bool):
            If True sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            abscissas:
                The quadrature points for where to evaluate the model function
                with ``abscissas.shape == (len(dist), N)`` where ``N`` is the
                number of samples.
            weights:
                The quadrature weights with ``weights.shape == (N,)``.

    Example:
        >>> abscissas, weights = quad_clenshaw_curtis(3, (0, 1))
        >>> abscissas.round(4)
        array([[0.  , 0.25, 0.75, 1.  ]])
        >>> weights.round(4)
        array([0.0556, 0.4444, 0.4444, 0.0556])
    """
    from ..distributions.baseclass import Dist
    if isinstance(domain, Dist):
        abscissas, weights = quad_clenshaw_curtis(
            order, (domain.lower, domain.upper), growth)
        weights *= domain.pdf(abscissas).flatten()
        weights /= numpy.sum(weights)
        return abscissas, weights

    order = numpy.asarray(order, dtype=int).flatten()
    lower, upper = numpy.array(domain)
    lower = numpy.asarray(lower).flatten()
    upper = numpy.asarray(upper).flatten()

    dim = max(lower.size, upper.size, order.size)

    order = numpy.ones(dim, dtype=int)*order
    lower = numpy.ones(dim)*lower
    upper = numpy.ones(dim)*upper

    if growth:
        order = numpy.where(order > 0, 2**order, 0)

    abscissas, weights = zip(*[_clenshaw_curtis(order_) for order_ in order])

    return combine_quadrature(abscissas, weights, (lower, upper))


def _clenshaw_curtis(order):
    r"""
    Backend method.

    Examples:
        >>> abscissas, weights = _clenshaw_curtis(0)
        >>> abscissas
        array([0.5])
        >>> weights
        array([1.])
        >>> abscissas, weights = _clenshaw_curtis(1)
        >>> abscissas
        array([0., 1.])
        >>> weights
        array([0.5, 0.5])
        >>> abscissas, weights = _clenshaw_curtis(2)
        >>> abscissas
        array([0. , 0.5, 1. ])
        >>> weights
        array([0.16666667, 0.66666667, 0.16666667])
        >>> abscissas, weights = _clenshaw_curtis(3)
        >>> abscissas
        array([0.  , 0.25, 0.75, 1.  ])
        >>> weights
        array([0.05555556, 0.44444444, 0.44444444, 0.05555556])
        >>> abscissas, weights = _clenshaw_curtis(4)
        >>> abscissas
        array([0.        , 0.14644661, 0.5       , 0.85355339, 1.        ])
        >>> weights
        array([0.03333333, 0.26666667, 0.4       , 0.26666667, 0.03333333])
    """
    if order == 0:
        return numpy.array([.5]), numpy.array([1.])

    theta = (order-numpy.arange(order+1))*numpy.pi/order
    abscisas = 0.5*numpy.cos(theta) + 0.5

    idx, idy = numpy.mgrid[:order+1, :order//2]
    weights = 2*numpy.cos(2*(idy+1)*theta[idx])/(4*idy*(idy+2)+3)
    if order % 2 == 0:
        weights[:, -1] *= 0.5
    weights = (1-numpy.sum(weights, -1)) / order

    weights[0] /= 2
    weights[-1] /= 2

    return abscisas, weights
