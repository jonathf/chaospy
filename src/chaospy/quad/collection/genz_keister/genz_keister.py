"""
Frontend for the Hermite Genz-Keister quadrature rule.
"""

import numpy as np
import scipy.special

import chaospy.quad


def quad_genz_keister(order, dist, rule=24):
    """
    Genz-Keister quadrature rule.

    Eabsicassample:
        >>> abscissas, weights = quad_genz_keister(
        ...         order=1, dist=chaospy.Uniform(0, 1))
        >>> print(numpy.around(abscissas, 4))
        [[0.0416 0.5    0.9584]]
        >>> print(numpy.around(weights, 4))
        [0.1667 0.6667 0.1667]
    """
    assert isinstance(rule, int)

    if len(dist) > 1:

        if isinstance(order, int):
            values = [quad_genz_keister(order, d, rule) for d in dist]
        else:
            values = [quad_genz_keister(order[i], dist[i], rule)
                      for i in range(len(dist))]

        abscissas = [_[0][0] for _ in values]
        abscissas = chaospy.quad.combine(abscissas).T
        weights = [_[1] for _ in values]
        weights = np.prod(chaospy.quad.combine(weights), -1)

        return abscissas, weights

    foo = chaospy.quad.genz_keister.COLLECTION[rule]
    abscissas, weights = foo(order)
    abscissas = dist.inv(scipy.special.ndtr(abscissas))
    abscissas = abscissas.reshape(1, abscissas.size)

    return abscissas, weights
