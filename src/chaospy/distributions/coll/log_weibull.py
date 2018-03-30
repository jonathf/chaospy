"""Gumbel or Log-Weibull distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add


class log_weibull(Dist):
    """Gumbel or Log-Weibull distribution."""

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        ex = numpy.exp(-x)
        return ex*numpy.exp(-ex)

    def _cdf(self, x):
        return numpy.exp(-numpy.exp(-x))

    def _ppf(self, q):
        return -numpy.log(-numpy.log(q))

    def _bnd(self):
        return self._ppf(1e-10), self._ppf(1-1e-10)


class LogWeibull(Add):
    """
    Gumbel or Log-Weibull distribution.

    Args:
        scale (float, Dist) : Scaling parameter
        loc (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.LogWeibull(2, 2)
        >>> print(distribution)
        LogWeibull(loc=2, scale=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [1.0482 2.1748 3.3435 4.9999]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.1609 0.1833 0.1532 0.0893]
        >>> print(numpy.around(distribution.sample(4), 4))
        [3.71   0.4572 7.952  2.631 ]
        >>> print(numpy.around(distribution.mom(1), 4))
        21.8892
    """
    def __init__(self, scale=1, loc=0):
        self._repr = {"scale": scale, "loc": loc}
        Add.__init__(self, left=log_weibull()*scale, right=loc)
