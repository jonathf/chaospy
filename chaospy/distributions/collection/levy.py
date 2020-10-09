"""Levy distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class levy(SimpleDistribution):
    """Levy distribution."""

    def __init__(self):
        super(levy, self).__init__()

    def _pdf(self, x):
        out = 1/numpy.sqrt(2*numpy.pi*x)/x*numpy.exp(-1/(2*x))
        out[x == 0] = 0
        return out

    def _cdf(self, x):
        return 2*(1-special.ndtr(1/numpy.sqrt(x)))

    def _ppf(self, q):
        val = special.ndtri(1-q/2.0)
        return 1.0/(val*val)

    def _upper(self):
        return 1e12

    def _lower(self):
        return 0.


class Levy(ShiftScaleDistribution):
    """
    Levy distribution

    Args:
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Levy()
        >>> distribution
        Levy()
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc[:5].round(3)
        array([ 0.   ,  0.609,  1.412,  3.636, 15.58 ])
        >>> distribution.upper
        array([1.e+12])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.369, 0.167, 0.05 , 0.006, 0.   ])
        >>> distribution.sample(4).round(3)
        array([  4.965,   0.403, 257.22 ,   2.025])

    """

    def __init__(self, scale=1, shift=0):
        super(Levy, self).__init__(
            dist=levy(),
            scale=scale,
            shift=shift,
        )
