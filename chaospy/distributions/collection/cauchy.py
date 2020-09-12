"""Cauchy distribution."""
import numpy

from ..baseclass import DistributionCore, ShiftScale


class cauchy(DistributionCore):
    """Standard Cauchy distribution."""

    def __init__(self):
        super(cauchy, self).__init__()

    def _pdf(self, x):
        return 1.0/numpy.pi/(1.0+x*x)

    def _cdf(self, x):
        return 0.5 + 1.0/numpy.pi*numpy.arctan(x)

    def _ppf(self, q):
        return numpy.tan(numpy.pi*q-numpy.pi/2.0)


class Cauchy(ShiftScale):
    """
    Cauchy distribution.

    Args:
        shift (float, Distribution):
            Location parameter
        scale (float, Distribution):
            Scaling parameter

    Examples:
        >>> distribution = chaospy.Cauchy(4, 2)
        >>> distribution
        Cauchy(scale=4, shift=2)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> distribution.inv(q).round(4)
        array([-4.9282, -0.3094,  2.    ,  4.3094,  8.9282])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.1667, 0.3333, 0.5   , 0.6667, 0.8333])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0199, 0.0597, 0.0796, 0.0597, 0.0199])
        >>> distribution.sample(4).round(4)
        array([ 4.0953, -8.585 , 27.4011,  1.776 ])
        >>> distribution.mom(1).round(4)
        2.0
    """

    def __init__(self, scale=1, shift=0):
        super(Cauchy, self).__init__(
            dist=cauchy(),
            scale=scale,
            shift=shift,
        )
