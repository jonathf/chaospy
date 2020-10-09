"""Gumbel or Log-Weibull distribution."""
import numpy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class log_weibull(SimpleDistribution):
    """Gumbel or Log-Weibull distribution."""

    def __init__(self):
        super(log_weibull, self).__init__()

    def _pdf(self, x):
        ex = numpy.exp(-x)
        return ex*numpy.exp(-ex)

    def _cdf(self, x):
        return numpy.exp(-numpy.exp(-x))

    def _ppf(self, q):
        return -numpy.log(-numpy.log(q))

    def _lower(self):
        return -3.5

    def _upper(self):
        return 35


class LogWeibull(ShiftScaleDistribution):
    """
    Gumbel or Log-Weibull distribution.

    Args:
        scale (float, Distribution):
            Scaling parameter
        loc (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.LogWeibull()
        >>> distribution
        LogWeibull()
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-3.5  , -0.476,  0.087,  0.672,  1.5  , 35.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.322, 0.367, 0.306, 0.179, 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.855, -0.771,  2.976,  0.316])

    """
    def __init__(self, scale=1, shift=0):
        super(LogWeibull, self).__init__(
            dist=log_weibull(),
            scale=scale,
            shift=shift,
        )
