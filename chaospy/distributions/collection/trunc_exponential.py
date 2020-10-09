"""Truncated exponential distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class truncexpon(SimpleDistribution):
    """Truncated exponential distribution."""

    def __init__(self, b):
        super(truncexpon, self).__init__(dict(b=b))

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


class TruncExponential(ShiftScaleDistribution):
    """
    Truncated exponential distribution.

    Args:
        upper (float, Distribution):
            Location of upper threshold
        scale (float, Distribution):
            Scaling parameter in the exponential distribution
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.TruncExponential(2, 4)
        >>> distribution
        TruncExponential(2, scale=4)
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
        super(TruncExponential, self).__init__(
            dist=truncexpon((upper-shift)*1./scale),
            scale=scale,
            shift=shift,
            repr_args=[upper],
        )
