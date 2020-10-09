"""Frechet or Extreme value distribution type 2."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class frechet(SimpleDistribution):
    """Frechet or Extreme value distribution type 2."""

    def __init__(self, c=1):
        super(frechet, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        return c*pow(x,c-1)*numpy.exp(-pow(x,c))

    def _cdf(self, x, c):
        return -numpy.expm1(-pow(x,c))

    def _ppf(self, q, c):
        return pow(-numpy.log1p(-q), 1./c)

    def _mom(self, k, c):
        return special.gamma(1-k*1./c)

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return pow(35, (1./c))


class Frechet(ShiftScaleDistribution):
    """
    Frechet or Extreme value distribution type 2.

    Args:
        shape (float, Distribution):
            Shape parameter.
        scale (float, Distribution):
            Scaling parameter.
        shift (float, Distribution):
            Location parameter.

    Examples:
        >>> distribution = chaospy.Frechet(3)
        >>> distribution
        Frechet(3)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.607, 0.799, 0.971, 1.172, 3.271])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.883, 1.15 , 1.132, 0.824, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.02 , 0.496, 1.442, 0.87 ])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(Frechet, self).__init__(
            dist=frechet(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
