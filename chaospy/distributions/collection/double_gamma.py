"""Double gamma distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class double_gamma(SimpleDistribution):
    """Double gamma distribution."""

    def __init__(self, a):
        super(double_gamma, self).__init__(dict(a=a))

    def _pdf(self, x, a):
        ax = abs(x)
        return 1.0/(2*special.gamma(a))*ax**(a-1.0) * numpy.exp(-ax)

    def _cdf(self, x, a):
        fac = 0.5*special.gammainc(a,abs(x))
        return numpy.where(x>0,0.5+fac,0.5-fac)

    def _ppf(self, q, a):
        fac = special.gammainccinv(a, 1-abs(2*q-1))
        out = numpy.where(q > 0.5, fac, -fac)
        return out

    def _lower(self, a):
        return -special.gammainccinv(a, 2e-15)

    def _upper(self, a):
        return special.gammainccinv(a, 2e-15)


class DoubleGamma(ShiftScaleDistribution):
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
        >>> distribution = chaospy.DoubleGamma(shape=1.5)
        >>> distribution
        DoubleGamma(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-35.769,  -1.473,  -0.503,   0.503,   1.473,  35.769])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.157, 0.242, 0.242, 0.157, 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.727, -2.154,  3.132, -0.138])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(DoubleGamma, self).__init__(
            dist=double_gamma(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
