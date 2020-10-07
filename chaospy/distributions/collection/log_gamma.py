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
        return numpy.log(special.gammaincinv(c,q))


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
        >>> distribution = chaospy.LogGamma(2, 2, 1)
        >>> distribution
        LogGamma(2, scale=2, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([0.6138, 1.639 , 2.4085, 3.1934])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.149 , 0.2392, 0.2706, 0.2245])
        >>> distribution.sample(4).round(4)
        array([ 2.6074, -0.0932,  4.1166,  1.9675])
        >>> distribution.mom(1).round(4)
        1.8456
    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(LogGamma, self).__init__(
            dist=log_gamma(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
