"""Pareto type 1 distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class pareto1(SimpleDistribution):
    """Pareto type 1 distribution."""

    def __init__(self, b):
        super(pareto1, self).__init__(dict(b=b))

    def _pdf(self, x, b):
        return b*x**(-b-1)

    def _cdf(self, x, b):
        return 1-x**-b

    def _ppf(self, q, b):
        return pow(1-q, -1./b)

    def _lower(self, b):
        return 1.

    def _upper(self, b):
        return pow(1e12, 1./b)


class Pareto1(ShiftScaleDistribution):
    """
    Pareto type 1 distribution.

    Lower threshold at scale+loc and survival: x^-shape

    Args:
        shape (float, Distribution):
            Tail index parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Pareto1(15)
        >>> distribution
        Pareto1(15)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([1.   , 1.015, 1.035, 1.063, 1.113, 6.31 ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([15.   , 11.823,  8.699,  5.644,  2.695,  0.   ])
        >>> distribution.sample(4).round(3)
        array([1.073, 1.008, 1.222, 1.045])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Pareto1, self).__init__(
            dist=pareto1(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
