# -*- coding: utf-8 -*-
"""
Fejér proposed two quadrature rules very similar to :ref:`clenshaw_curtis`.
The only difference is that the endpoints are set to zero. That is, Fejér only
used the interior extrema of the Chebyshev polynomials, i.e. the true
stationary points. This makes this a better method for performing quadrature on
infinite intervals, as the evaluation does not contain illegal values.

Example usage
-------------

The first few orders with linear growth rule::

    >>> distribution = chaospy.Uniform(0, 1)
    >>> for order in [0, 1, 2, 3]:
    ...     X, W = chaospy.generate_quadrature(
    ...         order, distribution, rule="fejer")
    ...     print(order, numpy.around(X, 3), numpy.around(W, 3))
    0 [[0.5]] [1.]
    1 [[0.25 0.75]] [0.5 0.5]
    2 [[0.146 0.5   0.854]] [0.286 0.429 0.286]
    3 [[0.095 0.345 0.655 0.905]] [0.188 0.312 0.312 0.188]

The first few orders with exponential growth rule::

    >>> for order in [0, 1, 2]:  # doctest: +NORMALIZE_WHITESPACE
    ...     X, W = chaospy.generate_quadrature(
    ...         order, distribution, rule="fejer", growth=True)
    ...     print(order, numpy.around(X, 2), numpy.around(W, 2))
    0 [[0.5]] [1.]
    1 [[0.15 0.5  0.85]] [0.29 0.43 0.29]
    2 [[0.04 0.15 0.31 0.5  0.69 0.85 0.96]]
        [0.07 0.14 0.18 0.2  0.18 0.14 0.07]

Applying the rule using Smolyak sparse grid::

    >>> distribution = chaospy.Iid(chaospy.Uniform(0, 1), 2)
    >>> X, W = chaospy.generate_quadrature(
    ...     2, distribution, rule="fejer", growth=True, sparse=True)
    >>> print(numpy.around(X, 3))  # doctest: +NORMALIZE_WHITESPACE
    [[0.038 0.146 0.146 0.146 0.309 0.5   0.5   0.5   0.5
      0.5   0.5   0.5   0.691 0.854 0.854 0.854 0.962]
     [0.5   0.146 0.5   0.854 0.5   0.038 0.146 0.309 0.5
      0.691 0.854 0.962 0.5   0.146 0.5   0.854 0.5  ]]
    >>> print(numpy.around(W, 3))  # doctest: +NORMALIZE_WHITESPACE
    [ 0.074  0.082 -0.021  0.082  0.184  0.074 -0.021  0.184 -0.273
      0.184 -0.021  0.074  0.184  0.082 -0.021  0.082  0.074]
"""
from __future__ import division, print_function

import numpy

from .combine import combine_quadrature


def quad_fejer(order, domain=(0, 1), growth=False):
    """
    Generate the quadrature abscissas and weights in Fejer quadrature.

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
        >>> abscissas, weights = quad_fejer(3, (0, 1))
        >>> print(numpy.around(abscissas, 4))
        [[0.0955 0.3455 0.6545 0.9045]]
        >>> print(numpy.around(weights, 4))
        [0.1804 0.2996 0.2996 0.1804]
    """
    from ..distributions.baseclass import Dist
    if isinstance(domain, Dist):
        abscissas, weights = quad_fejer(
            order, domain.range(), growth)
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
        order = numpy.where(order > 0, 2**(order+1)-2, 0)

    abscissas, weights = zip(*[_fejer(order_) for order_ in order])

    return combine_quadrature(abscissas, weights, (lower, upper))


def _fejer(order):
    r"""
    Backend method.

    Examples:
        >>> abscissas, weights = _fejer(0)
        >>> print(abscissas)
        [0.5]
        >>> print(weights)
        [1.]
        >>> abscissas, weights = _fejer(1)
        >>> print(abscissas)
        [0.25 0.75]
        >>> print(weights)
        [0.44444444 0.44444444]
        >>> abscissas, weights = _fejer(2)
        >>> print(abscissas)
        [0.14644661 0.5        0.85355339]
        >>> print(weights)
        [0.26666667 0.4        0.26666667]
        >>> abscissas, weights = _fejer(3)
        >>> print(abscissas)
        [0.0954915 0.3454915 0.6545085 0.9045085]
        >>> print(weights)
        [0.18037152 0.29962848 0.29962848 0.18037152]
        >>> abscissas, weights = _fejer(4)
        >>> print(abscissas)
        [0.0669873 0.25      0.5       0.75      0.9330127]
        >>> print(weights)
        [0.12698413 0.22857143 0.26031746 0.22857143 0.12698413]
        >>> abscissas, weights = _fejer(5)
        >>> print(abscissas)
        [0.04951557 0.1882551  0.38873953 0.61126047 0.8117449  0.95048443]
        >>> print(weights)
        [0.0950705  0.17612121 0.2186042  0.2186042  0.17612121 0.0950705 ]
    """
    order = int(order)
    if order == 0:
        return numpy.array([.5]), numpy.array([1.])

    order += 2

    theta = (order-numpy.arange(order+1))*numpy.pi/order
    abscisas = 0.5*numpy.cos(theta) + 0.5

    idx, idy = numpy.mgrid[:order+1, :order//2]
    weights = 2*numpy.cos(2*(idy+1)*theta[idx])/(4*idy*(idy+2)+3)
    if order % 2 == 0:
        weights[:, -1] *= 0.5
    weights = (1-numpy.sum(weights, -1)) / order

    return abscisas[1:-1], weights[1:-1]
