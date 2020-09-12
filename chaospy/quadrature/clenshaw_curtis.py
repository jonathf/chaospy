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
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

import numpy
import chaospy

from .combine import combine_quadrature


def quad_clenshaw_curtis(order, domain, growth=False, segments=1):
    """
    Generate the quadrature nodes and weights in Clenshaw-Curtis quadrature.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (chaospy.distributions.baseclass.Distribution, numpy.ndarray):
            Either distribution or bounding of interval to integrate over.
        growth (bool):
            If True sets the growth rule for the quadrature rule to only
            include orders that enhances nested samples.
        segments (int):
            Split intervals into N subintervals and create a patched
            quadrature based on the segmented quadrature. Can not be lower than
            `order`. If 0 is provided, default to square root of `order`.
            Nested samples only exist when the number of segments are fixed.

    Returns:
        (numpy.ndarray, numpy.ndarray):
            abscissas:
                The quadrature points for where to evaluate the model function
                with ``abscissas.shape == (len(dist), N)`` where ``N`` is the
                number of samples.
            weights:
                The quadrature weights with ``weights.shape == (N,)``.

    Example:
        >>> abscissas, weights = quad_clenshaw_curtis(4, (0, 1))
        >>> abscissas.round(4)
        array([[0.    , 0.1464, 0.5   , 0.8536, 1.    ]])
        >>> weights.round(4)
        array([0.0333, 0.2667, 0.4   , 0.2667, 0.0333])
        >>> abscissas, weights = quad_clenshaw_curtis(4, (0, 1), segments=0)
        >>> abscissas.round(4)
        array([[0.  , 0.25, 0.5 , 0.75, 1.  ]])
        >>> weights.round(4)
        array([0.0833, 0.3333, 0.1667, 0.3333, 0.0833])
    """
    if isinstance(domain, chaospy.Distribution):
        abscissas, weights = quad_clenshaw_curtis(
            order, (domain.lower, domain.upper), growth, segments)

        # Sometimes edge samples (inside the domain) falls out again from simple
        # rounding errors. Edge samples needs to be adjusted.
        eps = 1e-14*(domain.upper-domain.lower)
        abscissas_ = numpy.clip(abscissas.T, domain.lower+eps, domain.upper-eps).T
        weights *= domain.pdf(abscissas_).flatten()
        weights /= numpy.sum(weights)

        return abscissas, weights

    order = numpy.asarray(order, dtype=int).flatten()
    lower, upper = numpy.array(domain)
    lower = numpy.asarray(lower).flatten()
    upper = numpy.asarray(upper).flatten()

    dim = max(lower.size, upper.size, order.size)

    order = order*numpy.ones(dim, dtype=int)
    lower = lower*numpy.ones(dim)
    upper = upper*numpy.ones(dim)
    segments = segments*numpy.ones(dim, dtype=int)

    if growth:
        order = numpy.where(order > 0, 2**order, 0)

    abscissas, weights = zip(*[_clenshaw_curtis(order_, segment)
                               for order_, segment in zip(order, segments)])

    return combine_quadrature(abscissas, weights, (lower, upper))


@lru_cache(None)
def _clenshaw_curtis(order, segments=1):
    r"""
    Backend method.

    Examples:
        >>> abscissas, weights = _clenshaw_curtis(0, 0)
        >>> abscissas
        array([0.5])
        >>> weights
        array([1.])
        >>> abscissas, weights = _clenshaw_curtis(2, 0)
        >>> abscissas
        array([0. , 0.5, 1. ])
        >>> weights
        array([0.16666667, 0.66666667, 0.16666667])
        >>> abscissas, weights = _clenshaw_curtis(4, 0)
        >>> abscissas
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> weights
        array([0.08333333, 0.33333333, 0.16666667, 0.33333333, 0.08333333])
        >>> abscissas, weights = _clenshaw_curtis(8, 0)
        >>> abscissas.round(3)
        array([0.   , 0.073, 0.25 , 0.427, 0.5  , 0.573, 0.75 , 0.927, 1.   ])
        >>> weights.round(3)
        array([0.017, 0.133, 0.2  , 0.133, 0.033, 0.133, 0.2  , 0.133, 0.017])
        >>> abscissas, weights = _clenshaw_curtis(16, 0)
        >>> abscissas.round(3)
        array([0.   , 0.037, 0.125, 0.213, 0.25 , 0.287, 0.375, 0.463, 0.5  ,
               0.537, 0.625, 0.713, 0.75 , 0.787, 0.875, 0.963, 1.   ])
        >>> weights.round(3)
        array([0.008, 0.067, 0.1  , 0.067, 0.017, 0.067, 0.1  , 0.067, 0.017,
               0.067, 0.1  , 0.067, 0.017, 0.067, 0.1  , 0.067, 0.008])
    """
    if segments != 1 and order > 2:
        if not segments:
            segments = int(numpy.sqrt(order))
        assert segments < order, "few samples to distribute than intervals"
        abscissas = []
        weights = []

        nodes = numpy.linspace(0, 1, segments+1)
        for idx, (lower, upper) in enumerate(zip(nodes[:-1], nodes[1:])):

            order_ = order//segments + (idx < (order%segments))
            abscissa, weight = _clenshaw_curtis(order_, segments=1)
            abscissa = abscissa*(upper-lower) + lower
            weight = weight*(upper-lower)
            if abscissas:
                weights[-1] += weight[0]
                abscissa = abscissa[1:]
                weight = weight[1:]
            abscissas.extend(abscissa)
            weights.extend(weight)

        assert len(abscissas) == order+1, (len(abscissas), order+1)
        assert len(weights) == order+1
        return numpy.array(abscissas), numpy.array(weights)

    if order == 0:
        return numpy.array([.5]), numpy.array([1.])

    theta = (order-numpy.arange(order+1))*numpy.pi/order
    abscissas = 0.5*numpy.cos(theta) + 0.5

    idx, idy = numpy.mgrid[:order+1, :order//2]
    weights = 2*numpy.cos(2*(idy+1)*theta[idx])/(4*idy*(idy+2)+3)
    if order % 2 == 0:
        weights[:, -1] *= 0.5
    weights = (1-numpy.sum(weights, -1)) / order

    weights[0] /= 2
    weights[-1] /= 2

    assert len(abscissas) == order+1
    return abscissas, weights
