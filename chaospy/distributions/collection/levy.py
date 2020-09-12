"""Levy distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class levy(DistributionCore):
    """Levy distribution."""

    def __init__(self):
        super(levy, self).__init__()

    def _pdf(self, x):
        return 1/numpy.sqrt(2*numpy.pi*x)/x*numpy.exp(-1/(2*x))

    def _cdf(self, x):
        return 2*(1-special.ndtr(1/numpy.sqrt(x)))

    def _ppf(self, q):
        val = special.ndtri(1-q/2.0)
        return 1.0/(val*val)

    def _lower(self):
        return 0.


class Levy(ShiftScale):
    """
    Levy distribution

    Args:
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Levy(2, 2)
        >>> distribution
        Levy(scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([ 3.2177,  4.8236,  9.2728, 33.16  ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.1847, 0.0834, 0.0251, 0.0031])
        >>> distribution.sample(4).round(4)
        array([ 11.9303,   2.8051, 516.4406,   6.0494])
    """

    def __init__(self, scale=1, shift=0):
        super(Levy, self).__init__(
            dist=levy(),
            scale=scale,
            shift=shift,
        )
