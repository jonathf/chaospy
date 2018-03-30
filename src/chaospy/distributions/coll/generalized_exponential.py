"""Generalized exponential distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class generalized_exponential(Dist):
    """Generalized exponential distribution."""

    def __init__(self, a=1, b=1, c=1):
        Dist.__init__(self, a=a, b=b, c=c)

    def _pdf(self, x, a, b, c):
        return (a+b*(-numpy.expm1(-c*x)))*numpy.exp((-a-b)*x+b*(-numpy.expm1(-c*x))/c)

    def _cdf(self, x, a, b, c):
        return -numpy.expm1((-a-b)*x + b*(-numpy.expm1(-c*x))/c)

    def _bnd(self, a, b, c):
        return 0, 10**10


class GeneralizedExponential(Add):
    """
    Generalized exponential distribution.

    Args:
        a (float, Dist) : First shape parameter
        b (float, Dist) : Second shape parameter
        c (float, Dist) : Third shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Note:
        "An Extension of Marshall and Olkin's Bivariate Exponential Distribution",
        H.K. Ryu, Journal of the American Statistical Association, 1993.

        "The Exponential Distribution: Theory, Methods and Applications",
        N. Balakrishnan, Asit P. Basu.

    Examples:
        >>> distribution = chaospy.GeneralizedExponential(3, 2, 2, 2, 2)
        >>> print(distribution)
        GeneralizedExponential(a=3, b=2, c=2, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [2.1423 2.3113 2.5314 2.8774]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.3061 1.0605 0.7649 0.4168]
        >>> print(numpy.around(distribution.sample(4), 4))
        [2.6058 2.0794 3.5124 2.3933]
    """

    def __init__(self, a=1, b=1, c=1, scale=1, shift=0):
        self._repr = {"a": a, "b": b, "c": c, "scale": scale, "shift": shift}
        Add.__init__(
            self, left=generalized_exponential(a, b, c)*scale, right=shift)
