"""Pareto type 1 distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class pareto1(DistributionCore):
    """Pareto type 1 distribution."""

    def __init__(self, b):
        super(pareto1, self).__init__(b=b)

    def _pdf(self, x, b):
        return b * x**(-b-1)

    def _cdf(self, x, b):
        return 1 -  x**(-b)

    def _ppf(self, q, b):
        return pow(1-q, -1.0/b)

    def _lower(self, b):
        return 1.0


class Pareto1(ShiftScale):
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
        >>> distribution = chaospy.Pareto1(2, 2, 2)
        >>> distribution
        Pareto1(2, scale=2, shift=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([4.2361, 4.582 , 5.1623, 6.4721])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.7155, 0.4648, 0.253 , 0.0894])
        >>> distribution.sample(4).round(4)
        array([ 5.3981,  4.126 , 10.9697,  4.7794])
    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Pareto1, self).__init__(
            dist=pareto1(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
