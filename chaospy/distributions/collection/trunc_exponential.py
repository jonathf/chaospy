"""Truncated exponential distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class truncexpon(Dist):
    """Truncated exponential distribution."""

    def __init__(self, b):
        Dist.__init__(self, b=b)

    def _pdf(self, x, b):
        return numpy.exp(-x)/(1-numpy.exp(-b))

    def _cdf(self, x, b):
        return (1.0-numpy.exp(-x))/(1-numpy.exp(-b))

    def _ppf(self, q, b):
        return -numpy.log(1-q+q*numpy.exp(-b))

    def _lower(self, b):
        return 0.

    def _upper(self, b):
        return b


class TruncExponential(Add):
    """
    Truncated exponential distribution.

    Args:
        upper (float, Dist):
            Location of upper threshold
        scale (float, Dist):
            Scaling parameter in the exponential distribution
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.TruncExponential(2, 4)
        >>> distribution
        TruncExponential(scale=4, shift=0, upper=2)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([0.    , 0.4142, 0.8763, 1.3988, 2.    ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.6354, 0.5729, 0.5104, 0.4479, 0.3854])
        >>> distribution.sample(4).round(4)
        array([1.1891, 0.1852, 1.873 , 0.8415])
        >>> distribution.mom(1).round(4)
        0.917
    """

    def __init__(self, upper=1, scale=1, shift=0):
        self._repr = {"upper": upper, "scale": scale, "shift": shift}
        Add.__init__(
            self, left=truncexpon((upper-shift)*1./scale)*scale, right=shift)
