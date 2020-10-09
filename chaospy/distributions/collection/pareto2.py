"""Pareto type 2 distribution."""
from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class pareto2(SimpleDistribution):
    """Pareto type 2 distribution."""

    def __init__(self, c):
        super(pareto2, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return c*1.0/(1.0+x)**(c+1.0)

    def _cdf(self, x, c):
        return 1.0-1.0/(1.0+x)**c

    def _ppf(self, q, c):
        return pow(1.0-q, -1.0/c)-1

    def _lower(self, c):
        return 0.0

    def _upper(self, c):
        return pow(1e-12, -1./c)-1


class Pareto2(ShiftScaleDistribution):
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
        >>> distribution = chaospy.Pareto2(15)
        >>> distribution
        Pareto2(15)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.015, 0.035, 0.063, 0.113, 5.31 ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([15.   , 11.823,  8.699,  5.644,  2.695,  0.   ])
        >>> distribution.sample(4).round(3)
        array([0.073, 0.008, 0.222, 0.045])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Pareto2, self).__init__(
            dist=pareto2(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
