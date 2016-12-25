"""
Sparse grid constructor.

Method for turning collection of one dimensional quadrature rules into Smolyak
sparse grid tensor rules.
"""

import numpy
import scipy.misc

import chaospy.bertran


def sparse_grid(func, order, dim=None, skew=None):
    """
    Smolyak sparse grid constructor.

    Args:
        func (callable) : function that takes a single argument `order` of type
            `numpy` and with `order.shape = (dim,)`
        order (int, array_like) : The order of the grid. If `array_like`,
            it overrides both `dim` and `skew`.
        dim (int) : number of dimension.
        skew (list) : order skewness.
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
        weight *= (-1)**(order-sum(idb))*scipy.misc.comb(dim-1, order-sum(idb))
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
