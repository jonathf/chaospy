"""Gompertz distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class gompertz(SimpleDistribution):
    """Gompertz distribution."""

    def __init__(self, c):
        super(gompertz, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        ex = numpy.exp(x)
        return c*ex*numpy.exp(-c*(ex-1))

    def _cdf(self, x, c):
        return 1.0-numpy.exp(-c*(numpy.exp(x)-1))

    def _ppf(self, q, c):
        return numpy.log(1-1.0/c*numpy.log(1-q))

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return numpy.log(1+27.7/c)


class Gompertz(ShiftScaleDistribution):
    """
    Gompertz distribution

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Gompertz(1.5)
        >>> distribution
        Gompertz(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.139, 0.293, 0.477, 0.729, 2.969])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([1.5  , 1.379, 1.206, 0.967, 0.622, 0.   ])
        >>> distribution.sample(4).round(3)
        array([0.535, 0.078, 1.099, 0.364])

    """

    def __init__(self, shape, scale=1, shift=0):
        super(Gompertz, self).__init__(
            dist=gompertz(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
