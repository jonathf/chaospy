"""Log-gamma distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class log_gamma(SimpleDistribution):
    """Log-gamma distribution."""

    def __init__(self, c):
        super(log_gamma, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return numpy.exp(c*x-numpy.exp(x)-special.gammaln(c))

    def _cdf(self, x, c):
        return special.gammainc(c, numpy.exp(x))

    def _ppf(self, q, c):
        return numpy.log(special.gammaincinv(c, q))

    def _lower(self, c):
        return numpy.log(special.gammaincinv(c, 1e-15))

    def _upper(self, c):
        return numpy.log(special.gammaincinv(c, 1-1e-15))


class LogGamma(ShiftScaleDistribution):
    """
    Log-gamma distribution

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.LogGamma(1.5)
        >>> distribution
        LogGamma(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-22.836,  -0.688,  -0.068,   0.387,   0.842,   3.597])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.243, 0.4  , 0.462, 0.392, 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.503, -1.125,  1.364,  0.128])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(LogGamma, self).__init__(
            dist=log_gamma(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
