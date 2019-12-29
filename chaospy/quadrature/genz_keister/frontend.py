"""
Frontend for the Hermite Genz-Keister quadrature rule.
"""
import numpy
import scipy.special

from ..combine import combine
from .gk16 import quad_genz_keister_16
from .gk18 import quad_genz_keister_18
from .gk22 import quad_genz_keister_22
from .gk24 import quad_genz_keister_24

GENS_KEISTER_FUNCTIONS = {
    16: quad_genz_keister_16,
    18: quad_genz_keister_18,
    22: quad_genz_keister_22,
    24: quad_genz_keister_24,
}


def quad_genz_keister(order, dist, rule=24):
    """
    Genz-Keister quadrature rule.

    Eabsicassample:
        >>> abscissas, weights = quad_genz_keister(
        ...         order=1, dist=chaospy.Uniform(0, 1))
        >>> abscissas.round(4)
        array([[0.0416, 0.5   , 0.9584]])
        >>> weights.round(4)
        array([0.1667, 0.6667, 0.1667])
    """
    assert isinstance(rule, int)

    if len(dist) > 1:

        if isinstance(order, int):
            values = [quad_genz_keister(order, d, rule) for d in dist]
        else:
            values = [quad_genz_keister(order[i], dist[i], rule)
                      for i in range(len(dist))]

        abscissas = [_[0][0] for _ in values]
        abscissas = combine(abscissas).T
        weights = [_[1] for _ in values]
        weights = numpy.prod(combine(weights), -1)

        return abscissas, weights

    foo = GENS_KEISTER_FUNCTIONS[rule]
    abscissas, weights = foo(order)
    abscissas = dist.inv(scipy.special.ndtr(abscissas))
    abscissas = abscissas.reshape(1, abscissas.size)

    return abscissas, weights
