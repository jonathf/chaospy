"""
Clenshaw-Curtis quadrature method.
"""
from __future__ import division

import numpy
import chaospy.quad


def quad_clenshaw_curtis(
        order, lower=0, upper=1, growth=False, composite=1, part=None):
    """
    Generate the quadrature nodes and weights in Clenshaw-Curtis quadrature.

    Example:
        >>> abscissas, weights = quad_clenshaw_curtis(3, 0, 1)
        >>> print(abscissas)
        [[ 0.    0.25  0.75  1.  ]]
        >>> print(weights)
        [ 0.11111111  0.38888889  0.19444444  0.11111111]
    """
    order = numpy.asarray(order, dtype=int).flatten()
    lower = numpy.asarray(lower).flatten()
    upper = numpy.asarray(upper).flatten()

    dim = max(lower.size, upper.size, order.size)

    order = numpy.ones(dim, dtype=int)*order
    lower = numpy.ones(dim)*lower
    upper = numpy.ones(dim)*upper

    if isinstance(composite, int):
        composite = numpy.array([numpy.linspace(0, 1, composite+1)]*dim)

    else:
        composite = numpy.asarray(composite)
        if not composite.shape:
            composite = composite.flatten()
        if len(composite.shape) == 1:
            composite = numpy.array([composite])
        composite = ((composite.T-lower)/(upper-lower)).T

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
    """Backend method."""
    if order == 0:
        return numpy.array([.5]), numpy.array([1.])

    abscis = -numpy.cos(numpy.arange(order+1)*numpy.pi/order)
    abscis[numpy.abs(abscis) < 1e-14] = 0

    grid = numpy.meshgrid(*[numpy.arange(order//2+1)]*2)
    premat = 2./order*numpy.cos(2*grid[0]*grid[1]*numpy.pi/order)
    premat[:, 0] *= .5
    premat[:, -1] *= .5

    prevec = 2./(1-numpy.arange(0, order+1, 2)**2)
    prevec[0] *= .5
    prevec[-1] *= .5

    weight = numpy.dot(premat.T, prevec)
    weight = numpy.concatenate((weight, weight[-1-1*(order%2 == 0)::-1]))
    weight[order // 2] *= 2

    abscis = .5*abscis+.5
    weight *= .5

    length = len(abscis)

    if composite is None:
        composite = []
    composite = list(set(composite))
    composite = [c for c in composite if (c < 1) and (c > 0)]
    composite.sort()
    composite = [0] + composite + [1]

    abscissas = numpy.zeros((length-1)*(len(composite)-1)+1)
    weights = numpy.zeros((length-1)*(len(composite)-1)+1)
    for dim in range(len(composite)-1):
        abscissas[dim*length-dim:(dim+1)*length-dim] = \
                abscis*(composite[dim+1]-composite[dim]) + composite[dim]
        weights[dim*length-dim:(dim+1)*length-dim] += \
            weight*(composite[dim+1]-composite[dim])

    return abscissas, weights
