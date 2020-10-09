"""Hyperbolic secant distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class hyperbolic_secant(SimpleDistribution):
    """Hyperbolic secant distribution."""

    def __init__(self):
        super(hyperbolic_secant, self).__init__()

    def _pdf(self, x):
        return .5*numpy.cosh(numpy.pi*x/2.)**-1

    def _cdf(self, x):
        return 2/numpy.pi*numpy.arctan(numpy.e**(numpy.pi*x/2.))

    def _ppf(self, q):
        return 2/numpy.pi*numpy.log(numpy.tan(numpy.pi*q/2.))

    def _lower(self):
        return -21.7

    def _upper(self):
        return 21.7

    def _mom(self, k):
        return numpy.abs(special.euler(k.item())[-1])


class HyperbolicSecant(ShiftScaleDistribution):
    """
    Hyperbolic secant distribution

    Args:
        scale (float, Distribution):
            Scale parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.HyperbolicSecant()
        >>> distribution
        HyperbolicSecant()
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-21.7  ,  -0.716,  -0.203,   0.203,   0.716,  21.7  ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.294, 0.476, 0.476, 0.294, 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.32 , -1.082,  1.622, -0.036])
        >>> distribution.mom(2).round(3)
        1.0

    """

    def __init__(self, scale=1, shift=0):
        super(HyperbolicSecant, self).__init__(
            dist=hyperbolic_secant(),
            scale=scale,
            shift=shift,
        )
