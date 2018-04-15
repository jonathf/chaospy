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

    def _bnd(self, x, b):
        return 0.0, b


class TruncExponential(Add):
    """
    Truncated exponential distribution.

    Args:
        upper (float, Dist) : Location of upper threshold
        scale (float, Dist) : Scaling parameter in the exponential distribution
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.TruncExponential(2, 4)
        >>> print(distribution)
        TruncExponential(scale=4, shift=0, upper=2)
        >>> q = numpy.linspace(0, 1, 5)
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.     0.4142 0.8763 1.3988 2.    ]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.   0.25 0.5  0.75 1.  ]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.6354 0.5729 0.5104 0.4479 0.3854]
        >>> print(numpy.around(distribution.sample(4), 4))
        [1.1891 0.1852 1.873  0.8415]
        >>> print(numpy.around(distribution.mom(1), 4))
        1.0
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[1.0163 1.0024 1.0008]
         [0.3292 0.2671 0.2572]]
    """

    def __init__(self, upper=1, scale=1, shift=0):
        self._repr = {"upper": upper, "scale": scale, "shift": shift}
        Add.__init__(
            self, left=truncexpon((upper-shift)/scale)*scale, right=shift)
