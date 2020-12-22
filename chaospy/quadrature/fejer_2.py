# -*- coding: utf-8 -*-
"""Generate the quadrature abscissas and weights in Fejer quadrature."""

"""
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

import numpy
import chaospy

from .hypercube import hypercube_quadrature
from .clenshaw_curtis import _clenshaw_curtis


def quad_fejer_2(order, domain=(0, 1), growth=False, segments=1):
    """
    Generate the quadrature abscissas and weights in Fejér type II quadrature.

    Fejér proposed two quadrature rules very similar to
    :func:`chaospy.quad_clenshaw_curtis`. The only difference is that the
    endpoints are removed. That is, Fejér only used the interior extrema of the
    Chebyshev polynomials, i.e. the true stationary points. This makes this a
    better method for performing quadrature on infinite intervals, as the
    evaluation does not contain endpoint values.

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
        >>> abscissas, weights = quad_fejer_2(3, (0, 1))
        >>> abscissas.round(4)
        array([[0.0955, 0.3455, 0.6545, 0.9045]])
        >>> weights.round(4)
        array([0.1804, 0.2996, 0.2996, 0.1804])
        >>> abscissas, weights = quad_fejer_2(3, (0, 1), segments=2)
        >>> abscissas.round(4)
        array([[0.0732, 0.25  , 0.4268, 0.625 , 0.875 ]])
        >>> weights.round(4)
        array([0.1333, 0.2   , 0.1333, 0.2222, 0.2222])

    """
    order = numpy.asarray(order)
    order = numpy.where(growth, numpy.where(order > 0, 2**(order+1)-2, 0), order)
    return hypercube_quadrature(
        quad_func=_fejer_type_2,
        order=order,
        domain=domain,
        segments=segments,
    )


def _fejer_type_2(order):
    """
    Backend for Fejer type II quadrature.

    Same as Clenshaw-Curtis, but with the end nodes removed.
    """
    abscissas, weights = _clenshaw_curtis(order+2)
    return abscissas[1:-1], weights[1:-1]
