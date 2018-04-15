"""Double Weibull distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class double_weibull(Dist):
    """Double weibull distribution."""

    def __init__(self, c):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        ax = numpy.abs(x)
        Px = c/2.0*ax**(c-1.0)*numpy.exp(-ax**c)
        return Px

    def _cdf(self, x, c):
        Cx1 = 0.5*numpy.exp(-abs(x)**c)
        return numpy.where(x > 0, 1-Cx1, Cx1)

    def _ppf(self, q, c):
        q_ = numpy.where(q>.5, 1-q, q)
        c = c*numpy.ones(q_.shape)
        Cq1 = numpy.ones(q_.shape)
        indices = q_ == 0
        Cq1[indices] = 1e10
        indices = ~indices & (c != 0)
        Cq1[indices] = (-numpy.log(2*q_[indices]))**(1./c[indices])
        return numpy.where(q>.5, Cq1, -Cq1)

    def _bnd(self, x, c):
        return self._ppf(1e-10, c), self._ppf(1-1e-10, c)


class DoubleWeibull(Add):
    """
    Double Weibull distribution.

    Args:
        shape (float, Dist): Shape parameter
        scale (float, Dist): Scaling parameter
        shift (float, Dist): Location parameter

    Examples:
        >>> distribution = chaospy.DoubleWeibull(2, 4, 2)
        >>> print(distribution)
        DoubleWeibull(scale=4, shape=2, shift=2)
        >>> q = numpy.linspace(0, 1, 5)
        >>> print(numpy.around(distribution.inv(q), 4))
        [-16.903   -1.3302   2.       5.3302  20.903 ]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.   0.25 0.5  0.75 1.  ]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.     0.1041 0.     0.1041 0.    ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 4.4232 -2.8491  8.0772  1.2382]
        >>> print(numpy.around(distribution.mom(1), 4))
        2.0
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[ 2.      2.      2.    ]
         [16.0001 15.9407 32.1195]]
    """

    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=double_weibull(shape)*scale, right=shift)
