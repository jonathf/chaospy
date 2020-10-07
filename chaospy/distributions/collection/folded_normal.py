"""Folded normal distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class folded_normal(SimpleDistribution):
    """Folded normal distribution."""

    def __init__(self, c=1):
        super(folded_normal, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return numpy.sqrt(2.0/numpy.pi)*numpy.cosh(c*x)*numpy.exp(-(x*x+c*c)/2.0)

    def _cdf(self, x, c):
        return special.ndtr(x-c) + special.ndtr(x+c)-1.0

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return 8+c


class FoldedNormal(ShiftScaleDistribution):
    """
    Folded normal distribution.

    Args:
        mu (float, Distribution):
            Location parameter in normal distribution.
        scale (float, Distribution):
            Scaling parameter (in both normal and fold).
        shift (float, Distribution):
            Location of fold.

    Examples:
        >>> distribution = chaospy.FoldedNormal(3, 2, 1)
        >>> distribution
        FoldedNormal(3, scale=2, shift=1)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([3.3224, 4.4938, 5.5067, 6.6832])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.1417, 0.1934, 0.1932, 0.14  ])
        >>> distribution.sample(4).round(4)
        array([2.1633, 1.7828, 4.6912, 5.5157])
        >>> distribution.mom(1).round(4)
        5.034
    """

    def __init__(self, mu=0, scale=1, shift=0):
        super(FoldedNormal, self).__init__(
            dist=folded_normal(mu-shift),
            scale=scale,
            shift=shift,
            repr_args=[mu],
        )
