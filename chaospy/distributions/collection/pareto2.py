"""Pareto type 2 distribution."""
from ..baseclass import DistributionCore, ShiftScale


class pareto2(DistributionCore):
    """Pareto type 2 distribution."""

    def __init__(self, c):
        super(pareto2, self).__init__(c=c)

    def _pdf(self, x, c):
        return c*1.0/(1.0+x)**(c+1.0)

    def _cdf(self, x, c):
        return 1.0-1.0/(1.0+x)**c

    def _ppf(self, q, c):
        return pow(1.0-q,-1.0/c)-1

    def _lower(self, c):
        return 0.0


class Pareto2(ShiftScale):
    """
    Pareto type 2 distribution.

    Also known as Lomax distribution (for loc=0).

    Lower threshold at loc and survival: (1+x)^-shape.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        loc (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.Pareto2(2, 2, 2)
        >>> distribution
        Pareto2(2, scale=2, shift=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.2361, 2.582 , 3.1623, 4.4721])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.7155, 0.4648, 0.253 , 0.0894])
        >>> distribution.sample(4).round(4)
        array([3.3981, 2.126 , 8.9697, 2.7794])
    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Pareto2, self).__init__(
            dist=pareto2(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
