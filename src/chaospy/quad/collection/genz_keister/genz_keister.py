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
        >>> print(abscissas)
        [[ 0.04163226  0.5         0.95836774]]
        >>> print(weights)
        [ 0.16666667  0.66666667  0.16666667]
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
