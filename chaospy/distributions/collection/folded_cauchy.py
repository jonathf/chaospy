"""Folded Cauchy distribution."""
import numpy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class folded_cauchy(SimpleDistribution):
    """Folded Cauchy distribution."""

    def __init__(self, c=0):
        super(folded_cauchy, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return 1./(numpy.pi*(1+(x-c)**2))+1/(numpy.pi*(1+(x+c)**2))

    def _cdf(self, x, c):
        return (numpy.arctan(x-c)+numpy.arctan(x+c))/numpy.pi

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return 1e+16  # actually infinity


class FoldedCauchy(ShiftScaleDistribution):
    """
    Folded Cauchy distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.FoldedCauchy(3, 2, 1)
        >>> distribution
        FoldedCauchy(3, scale=2, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([ 5.1449,  6.708 ,  8.0077, 10.6502])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0915, 0.1603, 0.1306, 0.0393])
        >>> distribution.sample(4).round(4)
        array([ 3.2852,  3.988 ,  2.1266, 19.5435])

    """

    def __init__(self, shape=0, scale=1, shift=0):
        super(FoldedCauchy, self).__init__(
            dist=folded_cauchy(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
