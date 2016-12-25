"""
Gauss-Legendre quadrature rule.
"""
import numpy

import chaospy.quad


def quad_gauss_legendre(order, lower=0, upper=1, composite=None):
    """
    Generate the quadrature nodes and weights in Gauss-Legendre quadrature.

    Example:
        >>> abscissas, weights = quad_gauss_legendre(3)
        >>> print(abscissas)
        [[ 0.06943184  0.33000948  0.66999052  0.93056816]]
        >>> print(weights)
        [ 0.17392742  0.32607258  0.32607258  0.17392742]
    """
    order = numpy.asarray(order, dtype=int).flatten()
    lower = numpy.asarray(lower).flatten()
    upper = numpy.asarray(upper).flatten()

    dim = max(lower.size, upper.size, order.size)
    order = numpy.ones(dim, dtype=int)*order
    lower = numpy.ones(dim)*lower
    upper = numpy.ones(dim)*upper

    if composite is None:
        composite = numpy.array(0)
    composite = numpy.asarray(composite)

    if not composite.size:
        composite = numpy.array([numpy.linspace(0, 1, composite+1)]*dim)

    else:
        composite = numpy.array(composite)
        if len(composite.shape) <= 1:
            composite = numpy.transpose([composite])
        composite = ((composite.T-lower)/(upper-lower)).T

    results = [_gauss_legendre(order[i], composite[i]) for i in range(dim)]
    abscis = numpy.array([_[0] for _ in results])
    weights = numpy.array([_[1] for _ in results])

    abscis = chaospy.quad.combine(abscis)
    weights = chaospy.quad.combine(weights)

    abscis = (upper-lower)*abscis + lower
    weights = numpy.prod(weights*(upper-lower), 1)

    return abscis.T, weights


def _gauss_legendre(order, composite=1):
    """Backend function."""
    inner = numpy.ones(order+1)*0.5
    outer = numpy.arange(order+1)**2
    outer = outer/(16*outer-4.)

    banded = numpy.diag(numpy.sqrt(outer[1:]), k=-1) + numpy.diag(inner) + \
            numpy.diag(numpy.sqrt(outer[1:]), k=1)
    vals, vecs = numpy.linalg.eig(banded)

    abscis, weight = vals.real, vecs[0, :]**2
    indices = numpy.argsort(abscis)
    abscis, weight = abscis[indices], weight[indices]

    n_abscis = len(abscis)

    composite = numpy.array(composite).flatten()
    composite = list(set(composite))
    composite = [comp for comp in composite if (comp < 1) and (comp > 0)]
    composite.sort()
    composite = [0]+composite+[1]

    abscissas = numpy.empty(n_abscis*(len(composite)-1))
    weights = numpy.empty(n_abscis*(len(composite)-1))
    for dim in range(len(composite)-1):
        abscissas[dim*n_abscis:(dim+1)*n_abscis] = \
            abscis*(composite[dim+1]-composite[dim]) + composite[dim]
        weights[dim*n_abscis:(dim+1)*n_abscis] = \
            weight*(composite[dim+1]-composite[dim])

    return abscissas, weights
