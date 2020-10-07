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


class LogWeibull(ShiftScaleDistribution):
    """
    Gumbel or Log-Weibull distribution.

    Args:
        scale (float, Distribution):
            Scaling parameter
        loc (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.LogWeibull(2, 2)
        >>> distribution
        LogWeibull(scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
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
    def __init__(self, scale=1, shift=0):
        super(LogWeibull, self).__init__(
            dist=log_weibull(),
            scale=scale,
            shift=shift,
        )
