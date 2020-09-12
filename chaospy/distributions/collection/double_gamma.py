"""Double gamma distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class double_gamma(DistributionCore):
    """Double gamma distribution."""

    def __init__(self, a):
        super(double_gamma, self).__init__(a=a)

    def _pdf(self, x, a):
        ax = abs(x)
        return 1.0/(2*special.gamma(a))*ax**(a-1.0) * numpy.exp(-ax)

    def _cdf(self, x, a):
        fac = 0.5*special.gammainc(a,abs(x))
        return numpy.where(x>0,0.5+fac,0.5-fac)

    def _ppf(self, q, a):
        fac = special.gammainccinv(a,1-abs(2*q-1))
        return numpy.where(q>0.5, fac, -fac)


class DoubleGamma(ShiftScale):
    """
    Double gamma distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.DoubleGamma(2, 4, 2)
        >>> distribution
        DoubleGamma(2, scale=4, shift=2)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([-100.4566,   -4.7134,    2.    ,    8.7134,  104.4566])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.    , 0.0392, 0.    , 0.0392, 0.    ])
        >>> distribution.sample(4).round(4)
        array([ 6.4679, -9.2251, 17.5874,  0.8239])
        >>> distribution.mom(1).round(4)
        2.0
    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(DoubleGamma, self).__init__(
            dist=double_gamma(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
