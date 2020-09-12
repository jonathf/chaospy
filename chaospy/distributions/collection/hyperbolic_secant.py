"""Hyperbolic secant distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class hyperbolic_secant(DistributionCore):
    """Hyperbolic secant distribution."""

    def __init__(self):
        super(hyperbolic_secant, self).__init__()

    def _pdf(self, x):
        return .5*numpy.cosh(numpy.pi*x/2.)**-1

    def _cdf(self, x):
        return 2/numpy.pi*numpy.arctan(numpy.e**(numpy.pi*x/2.))

    def _ppf(self, q):
        return 2/numpy.pi*numpy.log(numpy.tan(numpy.pi*q/2.))

    def _mom(self, k):
        shape = k.shape
        output = numpy.abs([special.euler(k_)[-1] for k_ in k.flatten()])
        return output.reshape(shape)


class HyperbolicSecant(ShiftScale):
    """
    Hyperbolic secant distribution

    Args:
        scale (float, Distribution):
            Scale parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.HyperbolicSecant(2, 2)
        >>> distribution
        HyperbolicSecant(scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([0.5687, 1.5933, 2.4067, 3.4313])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.1469, 0.2378, 0.2378, 0.1469])
        >>> distribution.sample(4).round(4)
        array([ 2.6397, -0.1648,  5.2439,  1.9287])
        >>> distribution.mom(1).round(4)
        2.0
    """

    def __init__(self, scale=1, shift=0):
        super(HyperbolicSecant, self).__init__(
            dist=hyperbolic_secant(),
            scale=scale,
            shift=shift,
        )
