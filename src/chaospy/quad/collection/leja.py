"""
Laja quadrature.

After paper by Narayan and Jakeman.
"""
import numpy
from scipy.optimize import fminbound

import chaospy.quad


def quad_leja(order, dist):
    """
    Generate Leja quadrature node.

    Example:
        >>> abscisas, weights = quad_leja(3, chaospy.Normal(0, 1))
        >>> print(abscisas)
        [[-2.71725575 -1.41421279  0.          1.76349491]]
        >>> print(weights)
        [ 0.02201589  0.16287411  0.65057226  0.16453774]
    """
    assert not dist.dependent()

    if len(dist) > 1:
        if isinstance(order, int):
            out = [quad_leja(order, _) for _ in dist]
        else:
            out = [quad_leja(order[_], dist[_]) for _ in range(len(dist))]

        abscissas = [_[0][0] for _ in out]
        weights = [_[1] for _ in out]
        abscissas = chaospy.quad.combine(abscissas).T
        weights = chaospy.quad.combine(weights)
        weights = numpy.prod(weights, -1)

        return abscissas, weights

    lower, upper = dist.range()
    abscissas = [lower, dist.mom(1), upper]
    for _ in range(order):

        obj = create_objective(dist, abscissas)
        opts, vals = zip(
            *[fminbound(
                obj, abscissas[idx], abscissas[idx+1], full_output=1)[:2]
              for idx in range(len(abscissas)-1)]
        )
        index = numpy.argmin(vals)
        abscissas.insert(index+1, opts[index])

    abscissas = numpy.asfarray(abscissas).flatten()[1:-1]
    weights = create_weights(abscissas, dist)
    abscissas = abscissas.reshape(1, abscissas.size)

    return numpy.array(abscissas), numpy.array(weights)


def create_objective(dist, abscissas):
    """Create objective function."""
    abscissas_ = numpy.array(abscissas[1:-1])
    def obj(absisa):
        """Local objective function."""
        out = -numpy.sqrt(dist.pdf(absisa))
        out *= numpy.prod(numpy.abs(abscissas_ - absisa))
        return out
    return obj


def create_weights(nodes, dist):
    """Create weights for the Laja method."""
    poly = chaospy.quad.generate_stieltjes(dist, len(nodes)-1, retall=True)[0]
    poly = chaospy.poly.flatten(chaospy.poly.Poly(poly))
    weights_inverse = poly(nodes)
    weights = numpy.linalg.inv(weights_inverse)
    return weights[:, 0]
