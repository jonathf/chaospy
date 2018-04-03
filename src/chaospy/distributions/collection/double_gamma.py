"""Double gamma distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class double_gamma(Dist):
    """Double gamma distribution."""

    def __init__(self, a):
        Dist.__init__(self, a=a)

    def _pdf(self, x, a):
        ax = abs(x)
        return 1.0/(2*special.gamma(a))*ax**(a-1.0) * numpy.exp(-ax)

    def _cdf(self, x, a):
        fac = 0.5*special.gammainc(a,abs(x))
        return numpy.where(x>0,0.5+fac,0.5-fac)

    def _ppf(self, q, a):
        fac = special.gammainccinv(a,1-abs(2*q-1))
        return numpy.where(q>0.5, fac, -fac)

    def _bnd(self, x, a):
        return self._ppf(1e-10, a), self._ppf(1-1e-10, a)


class DoubleGamma(Add):
    """
    Double gamma distribution.

    Args:
        shape (float, Dist): Shape parameter
        scale (float, Dist): Scaling parameter
        shift (float, Dist): Location parameter

    Examples:
        >>> distribution = chaospy.DoubleGamma(2, 4, 2)
        >>> print(distribution)
        DoubleGamma(scale=4, shape=2, shift=2)
        >>> q = numpy.linspace(0, 1, 5)
        >>> print(numpy.around(distribution.inv(q), 4))
        [-100.4566   -4.7134    2.        8.7134  104.4566]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.   0.25 0.5  0.75 1.  ]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.     0.0392 0.     0.0392 0.    ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 6.4679 -9.2251 17.5874  0.8239]
        >>> print(numpy.around(distribution.mom(1), 4))
        2.0
    """

    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=double_gamma(shape)*scale, right=shift)
