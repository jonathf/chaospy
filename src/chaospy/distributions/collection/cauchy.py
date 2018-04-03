"""Cauchy distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add


class cauchy(Dist):
    """Standard Cauchy distribution."""

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return 1.0/numpy.pi/(1.0+x*x)

    def _cdf(self, x):
        return 0.5 + 1.0/numpy.pi*numpy.arctan(x)

    def _ppf(self, q):
        return numpy.tan(numpy.pi*q-numpy.pi/2.0)

    def _bnd(self, x):
        return self._ppf(1e-10), self._ppf(1-1e-10)


class Cauchy(Add):
    """
    Cauchy distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter

    Examples:
        >>> distribution = chaospy.Cauchy(2, 4)
        >>> print(distribution)
        Cauchy(loc=2, scale=4)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [-4.9282 -0.3094  2.      4.3094  8.9282]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.0199 0.0597 0.0796 0.0597 0.0199]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 4.0953 -8.585  27.4011  1.776 ]
    """

    def __init__(self, loc=0, scale=1):
        self._repr = {"loc": loc, "scale": scale}
        Add.__init__(self, left=cauchy()*scale, right=loc)
