"""
As the number of dimensions increases linear, the number of samples increases
exponentially. This is known as the curse of dimensionality. Except for
switching to Monte Carlo integration, the is no way to completly guard against
this problem. However, there are some possibility to mitigate the problem
personally. One such strategy is to employ Smolyak sparse-grid quadrature. This
method uses a quadrature rule over a combination of different orders to tailor
a scheme that uses fewer abscissas points than a full tensor-product approach.

To use Smolyak sparse-grid in ``chaospy``, just pass the flag ``sparse=True``
to the ``generate_quadrature`` function. For example::

    >>> distribution = chaospy.J(chaospy.Uniform(0, 4), chaospy.Uniform(0, 4))
    >>> X, W = chaospy.generate_quadrature(3, distribution, sparse=True)
    >>> print(numpy.around(X, 4))
    [[0. 0. 0. 1. 2. 2. 2. 2. 2. 3. 4. 4. 4.]
     [0. 2. 4. 2. 0. 1. 2. 3. 4. 2. 0. 2. 4.]]
    >>> print(numpy.around(W, 4))
    [-0.0833  0.2222 -0.0833  0.4444  0.2222  0.4444 -1.3333  0.4444  0.2222
      0.4444 -0.0833  0.2222 -0.0833]

This compared to the full tensor-product grid::

    >>> X, W = chaospy.generate_quadrature(3, distribution, sparse=False)
    >>> print(numpy.around(X, 4))
    [[0. 0. 0. 0. 1. 1. 1. 1. 3. 3. 3. 3. 4. 4. 4. 4.]
     [0. 1. 3. 4. 0. 1. 3. 4. 0. 1. 3. 4. 0. 1. 3. 4.]]
    >>> print(numpy.around(W, 4))
    [0.0031 0.0247 0.0247 0.0031 0.0247 0.1975 0.1975 0.0247 0.0247 0.1975
     0.1975 0.0247 0.0031 0.0247 0.0247 0.0031]

The method works with all quadrature rules, but is known to be quite
inefficient when applied to rules that can not be nested. For example using
Gauss-Legendre samples::

    >>> X, W = chaospy.generate_quadrature(
    ...     6, distribution, rule="gauss_legendre", sparse=True)
    >>> print(len(W))
    140
    >>> X, W = chaospy.generate_quadrature(
    ...     6, distribution, rule="gauss_legendre", sparse=False)
    >>> print(len(W))
    49

.. note:
    Some quadrature rules are only partially nested at certain orders. These
    include e.g. :ref:`clenshaw_curtis`, :ref:`fejer` and :ref:`newton_cotes`.
    To exploit this nested-nes, the default behavior is to only include orders
    that are properly nested. This implies that flipping the flag ``sparse``
    will result in a somewhat different scheme. To fix the scheme one way or
    the other, explicitly include the flag ``growth=False`` or ``growth=True``
    respectively.
"""
from collections import defaultdict
from itertools import product

import numpy
from scipy.special import comb

from ..bertran import bindex

from .interface import construct_quadrature


def sparse_grid(
        order,
        dist,
        accuracy=100,
        rule="G",
        growth=None,
):
    """
    Smolyak sparse grid constructor.

    Args:
        func (:py:data:typing.Callable):
            Function that takes a single argument ``order`` of type
            ``numpy.ndarray`` and with ``order.shape = (dim,)``
        order (int, numpy.ndarray):
            The order of the grid. If ``numpy.ndarray``, it overrides both
            ``dim`` and ``skew``.
        dim (int):
            Number of dimension.

    Example:
        >>> distribution = chaospy.J(chaospy.Normal(0, 1), chaospy.Uniform(-1, 1))
        >>> X, W = sparse_grid(1, distribution)
        >>> print(numpy.around(X, 4))
        [[-1.      0.      0.      0.      1.    ]
         [ 0.     -0.5774  0.      0.5774  0.    ]]
        >>> print(numpy.around(W, 4))
        [ 0.5  0.5 -1.   0.5  0.5]
        >>> X, W = sparse_grid([2, 1], distribution)
        >>> print(numpy.around(X, 3))
        [[-1.732 -1.    -1.    -1.     0.     1.     1.     1.     1.732]
         [ 0.    -0.577  0.     0.577  0.    -0.577  0.     0.577  0.   ]]
        >>> print(numpy.around(W, 3))
        [ 0.167  0.25  -0.5    0.25   0.667  0.25  -0.5    0.25   0.167]
    """
    orders = order*numpy.ones(len(dist), dtype=int)
    order = numpy.min(orders)
    skew = orders-order

    if isinstance(rule, str):
        rule = (rule,)*len(dist)

    # Create a quick look-up table so values do not need to be re-calculatated
    # on the fly.
    abscissas_ = [[] for _ in range(len(dist))]
    weights_ = [[] for _ in range(len(dist))]
    for idx, (order_, dist_, rule_) in enumerate(zip(orders, dist, rule)):
        for idy in range(order_+1):
            (abscissas,), weights = construct_quadrature(
                idy, dist_, accuracy=accuracy, rule=rule_, growth=growth)
            abscissas_[idx].append(abscissas)
            weights_[idx].append(weights)

    # Indices and coefficients used in the calculations
    indices = bindex(order-len(dist)+1, order, dim=len(dist))
    coeffs = numpy.sum(indices, -1)
    coeffs = (2*((order-coeffs+1) % 2)-1)*comb(len(dist)-1, order-coeffs)

    collection = defaultdict(float)
    for bidx, coeff in zip(indices+skew, coeffs.tolist()):
        xs = [xw[idx] for idx, xw in zip(bidx, abscissas_)]
        ws = [xw[idx] for idx, xw in zip(bidx, weights_)]
        for x, w in zip(product(*xs), product(*ws)):
            collection[x] += numpy.prod(w)*coeff

    abscissas = sorted(collection)
    weights = numpy.array([collection[key] for key in abscissas])
    abscissas = numpy.array(abscissas).T
    return abscissas, weights
