# -*- coding: utf-8 -*-
"""
Generate the quadrature abscissas and weights in Fejer quadrature.

Example usage
-------------

The first few orders with linear growth rule::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> for order in [0, 1, 2, 3]:
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="fejer")
    ...     print(order, abscissas.round(3), weights.round(3))
    0 [[0.5]] [1.]
    1 [[0.25 0.75]] [0.5 0.5]
    2 [[0.146 0.5   0.854]] [0.286 0.429 0.286]
    3 [[0.095 0.345 0.655 0.905]] [0.188 0.312 0.312 0.188]

The first few orders with exponential growth rule::

    >>> for order in [0, 1, 2]:  # doctest: +NORMALIZE_WHITESPACE
    ...     abscissas, weights = chaospy.generate_quadrature(
    ...         order, distribution, rule="fejer", growth=True)
    ...     print(order, abscissas.round(2), weights.round(2))
    0 [[0.5]] [1.]
    1 [[0.15 0.5  0.85]] [0.29 0.43 0.29]
    2 [[0.04 0.15 0.31 0.5  0.69 0.85 0.96]]
        [0.07 0.14 0.18 0.2  0.18 0.14 0.07]

Applying the rule using Smolyak sparse grid::

    >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    >>> abscissas, weights = chaospy.generate_quadrature(
    ...     2, distribution, rule="fejer", growth=True, sparse=True)
    >>> abscissas.round(3)
    array([[0.038, 0.146, 0.146, 0.146, 0.309, 0.5  , 0.5  , 0.5  , 0.5  ,
            0.5  , 0.5  , 0.5  , 0.691, 0.854, 0.854, 0.854, 0.962],
           [0.5  , 0.146, 0.5  , 0.854, 0.5  , 0.038, 0.146, 0.309, 0.5  ,
            0.691, 0.854, 0.962, 0.5  , 0.146, 0.5  , 0.854, 0.5  ]])
    >>> weights.round(3)
    array([ 0.074,  0.082, -0.021,  0.082,  0.184,  0.074, -0.021,  0.184,
           -0.273,  0.184, -0.021,  0.074,  0.184,  0.082, -0.021,  0.082,
            0.074])
"""
from __future__ import division
try:
    from functools import lru_cache
except ImportError:  # pragma: no cover
    from functools32 import lru_cache

import numpy
import chaospy

from .combine import combine_quadrature
from .clenshaw_curtis import _clenshaw_curtis


def quad_fejer(order, domain=(0, 1), growth=False, segments=1):
    """
    Generate the quadrature abscissas and weights in Fejér quadrature.

    Fejér proposed two quadrature rules very similar to
    :func:`quad_clenshaw_curtis`. The only difference is that the endpoints are
    removed. That is, Fejér only used the interior extrema of the Chebyshev
    polynomials, i.e. the true stationary points. This makes this a better
    method for performing quadrature on infinite intervals, as the evaluation
    does not contain illegal values.

    Args:
        order (int, numpy.ndarray):
            Quadrature order.
        domain (chaospy.Distribution, numpy.ndarray):
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
        abscissas (numpy.ndarray):
            The quadrature points for where to evaluate the model function
            with ``abscissas.shape == (len(dist), N)`` where ``N`` is the
            number of samples.
        weights (numpy.ndarray):
            The quadrature weights with ``weights.shape == (N,)``.

    Notes:
        Implemented as proposed by Waldvogel :cite:`waldvogel_fast_2006`.

    Example:
        >>> abscissas, weights = quad_fejer(3, (0, 1))
        >>> abscissas.round(4)
        array([[0.0955, 0.3455, 0.6545, 0.9045]])
        >>> weights.round(4)
        array([0.1804, 0.2996, 0.2996, 0.1804])
        >>> abscissas, weights = quad_fejer(3, (0, 1), segments=2)
        >>> abscissas.round(4)
        array([[0.125, 0.375, 0.625, 0.875]])
        >>> weights.round(4)
        array([0.2222, 0.2222, 0.2222, 0.2222])
    """
    if isinstance(domain, chaospy.Distribution):
        abscissas, weights = quad_fejer(
            order, (domain.lower, domain.upper), growth)
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
        order = numpy.where(order > 0, 2**(order+1)-2, 0)

    abscissas, weights = zip(*[_fejer(order_, segment)
                               for order_, segment in zip(order, segments)])

    return combine_quadrature(abscissas, weights, (lower, upper))


@lru_cache(None)
def _fejer(order, segments=1):
    """Backend method."""
    if segments != 1 and order > 2:
        if not segments:
            segments = int(numpy.sqrt(order))
        assert segments < order, "few samples to distribute than intervals"
        abscissas = []
        weights = []

        nodes = numpy.linspace(0, 1, segments+1)
        for idx, (lower, upper) in enumerate(zip(nodes[:-1], nodes[1:])):
            order_ = order//segments + (idx+1 < (order%segments))
            abscissa, weight = _fejer(order_, segments=1)
            abscissas.extend(abscissa*(upper-lower) + lower)
            weights.extend(weight*(upper-lower))

        assert len(abscissas) == order+1, (len(abscissas), order+1)
        assert len(weights) == order+1
        return numpy.array(abscissas), numpy.array(weights)

    abscissas, weights = _clenshaw_curtis(order+2, segments)
    return abscissas[1:-1], weights[1:-1]
