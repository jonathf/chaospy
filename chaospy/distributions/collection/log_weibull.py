"""Gumbel or Log-Weibull distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add
from .deprecate import deprecation_warning


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


class LogWeibull(Add):
    """
    Gumbel or Log-Weibull distribution.

    Args:
        scale (float, Dist):
            Scaling parameter
        loc (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.LogWeibull(2, 2)
        >>> distribution
        LogWeibull(loc=2, scale=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([1.0482, 2.1748, 3.3435, 4.9999])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.1609, 0.1833, 0.1532, 0.0893])
        >>> distribution.sample(4).round(4)
        array([3.71  , 0.4572, 7.952 , 2.631 ])
        >>> distribution.mom(1).round(4)
        3.1544
    """
    def __init__(self, scale=1, loc=0):
        self._repr = {"scale": scale, "loc": loc}
        Add.__init__(self, left=log_weibull()*scale, right=loc)


Logweibull = deprecation_warning(LogWeibull, "Logweibull")
