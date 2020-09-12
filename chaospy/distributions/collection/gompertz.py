"""Gompertz distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class gompertz(DistributionCore):
    """Gompertz distribution."""

    def __init__(self, c):
        super(gompertz, self).__init__(c=c)

    def _pdf(self, x, c):
        ex = numpy.exp(x)
        return c*ex*numpy.exp(-c*(ex-1))

    def _cdf(self, x, c):
        return 1.0-numpy.exp(-c*(numpy.exp(x)-1))

    def _ppf(self, q, c):
        return numpy.log(1-1.0/c*numpy.log(1-q))

    def _lower(self, c):
        return 0.


class Gompertz(ShiftScale):
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
        >>> distribution = chaospy.Gompertz(3, 2, 2)
        >>> distribution
        Gompertz(3, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.1435, 2.3145, 2.5331, 2.859 ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([1.2893, 1.0532, 0.7833, 0.4609])
        >>> distribution.sample(4).round(4)
        array([2.6052, 2.0798, 3.3868, 2.3967])
        >>> distribution.mom(1).round(4)
        2.5242
    """

    def __init__(self, shape, scale, shift):
        super(Gompertz, self).__init__(
            dist=gompertz(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
