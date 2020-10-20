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
        return special.ndtr(x-c)+special.ndtr(x+c)-1.0

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
        >>> distribution = chaospy.FoldedNormal(1.5)
        >>> distribution
        FoldedNormal(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.706, 1.254, 1.755, 2.342, 9.5  ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.259, 0.326, 0.396, 0.388, 0.28 , 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.906, 2.225, 1.638, 2.701])

    """

    def __init__(self, mu=0, scale=1, shift=0):
        super(FoldedNormal, self).__init__(
            dist=folded_normal(mu-shift),
            scale=scale,
            shift=shift,
            repr_args=[mu],
        )
