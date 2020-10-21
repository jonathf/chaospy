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
        >>> distribution = chaospy.TruncExponential(1.5)
        >>> distribution
        TruncExponential(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.169, 0.372, 0.628, 0.972, 1.5  ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([1.287, 1.087, 0.887, 0.687, 0.487, 0.287])
        >>> distribution.sample(4).round(3)
        array([0.709, 0.094, 1.34 , 0.469])

    """

    def __init__(self, upper=1, scale=1, shift=0):
        super(TruncExponential, self).__init__(
            dist=truncexpon((upper-shift)*1./scale),
            scale=scale,
            shift=shift,
            repr_args=[upper],
        )
