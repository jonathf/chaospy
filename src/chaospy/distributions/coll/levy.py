"""Levy distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class levy(Dist):
    """Levy distribution."""

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return 1/numpy.sqrt(2*numpy.pi*x)/x*numpy.exp(-1/(2*x))

    def _cdf(self, x):
        return 2*(1-special.ndtr(1/numpy.sqrt(x)))

    def _ppf(self, q):
        val = special.ndtri(1-q/2.0)
        return 1.0/(val*val)

    def _bnd(self):
        return 0., self._ppf(1-1e-10)


class Levy(Add):
    """
    Levy distribution

    Args:
        loc (float, Dist): Location parameter
        scale (float, Dist): Scaling parameter

    Examples:
        >>> distribution = chaospy.Levy(2, 2)
        >>> print(distribution)
        Levy(loc=2, scale=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [ 3.2177  4.8236  9.2728 33.16  ]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.1847 0.0834 0.0251 0.0031]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 11.9303   2.8051 516.4406   6.0494]
        >>> print(distribution.mom(1) > 1e10) # undefined
        True
    """
    def __init__(self, loc=0, scale=1):
        self._repr = {"loc": loc, "scale": scale}
        Add.__init__(self, left=levy()*scale, right=loc)
