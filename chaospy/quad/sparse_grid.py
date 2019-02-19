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
    >>> X, W = chaospy.generate_quadrature(3, distribution, normalize=True, sparse=True)
    >>> print(numpy.around(X, 4))
    [[0. 2. 4. 2. 0. 1. 2. 3. 4. 2. 0. 2. 4.]
     [0. 0. 0. 1. 2. 2. 2. 2. 2. 3. 4. 4. 4.]]
    >>> print(numpy.around(W, 4))
    [-0.0833  0.2222 -0.0833  0.4444  0.2222  0.4444 -0.6667  0.4444  0.2222
      0.4444 -0.0833  0.2222 -0.0833]

This compared to the full tensor-product grid::

    >>> X, W = chaospy.generate_quadrature(3, distribution, normalize=True)
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
    ...     3, distribution, rule="E", sparse=True)
    >>> print(len(W))
    119
    >>> X, W = chaospy.generate_quadrature(
    ...     3, distribution, rule="E", sparse=False)
    >>> print(len(W))
    64
"""

import numpy
from scipy.special import comb

import chaospy.bertran


def sparse_grid(func, order, dim=None, skew=None):
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
        skew (list):
            Order skewness.
    """
    if not isinstance(order, int):
        orders = numpy.array(order).flatten()
        dim = orders.size
        m_order = int(numpy.min(orders))
        skew = [order-m_order for order in orders]
        return sparse_grid(func, m_order, dim, skew)

    abscissas, weights = [], []
    bindex = chaospy.bertran.bindex(order-dim+1, order, dim)

    if skew is None:
        skew = numpy.zeros(dim, dtype=int)
    else:
        skew = numpy.array(skew, dtype=int)
        assert len(skew) == dim

    for idx in range(
            chaospy.bertran.terms(order, dim)
            - chaospy.bertran.terms(order-dim, dim)):

        idb = bindex[idx]
        abscissa, weight = func(skew+idb)
        weight *= (-1)**(order-sum(idb))*comb(dim-1, order-sum(idb))
        abscissas.append(abscissa)
        weights.append(weight)

    abscissas = numpy.concatenate(abscissas, 1)
    weights = numpy.concatenate(weights, 0)

    abscissas = numpy.around(abscissas, 15)
    order = numpy.lexsort(tuple(abscissas))
    abscissas = abscissas.T[order].T
    weights = weights[order]

    # identify non-unique terms
    diff = numpy.diff(abscissas.T, axis=0)
    unique = numpy.ones(len(abscissas.T), bool)
    unique[1:] = (diff != 0).any(axis=1)

    # merge duplicate nodes
    length = len(weights)
    idx = 1
    while idx < length:
        while idx < length and unique[idx]:
            idx += 1
        idy = idx+1
        while idy < length and not unique[idy]:
            idy += 1
        if idy-idx > 1:
            weights[idx-1] = numpy.sum(weights[idx-1:idy])
        idx = idy+1

    abscissas = abscissas[:, unique]
    weights = weights[unique]

    return abscissas, weights
