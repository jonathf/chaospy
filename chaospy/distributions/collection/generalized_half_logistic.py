"""Generalized half-logistic distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class generalized_half_logistic(SimpleDistribution):
    """Generalized half-logistic distribution."""

    def __init__(self, c=1):
        super(generalized_half_logistic, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        limit = 1.0/c
        tmp = (1-c*x)
        tmp0 = tmp**(limit-1)
        tmp2 = tmp0*tmp
        return 2*tmp0/(1+tmp2)**2

    def _cdf(self, x, c):
        limit = 1.0/c
        tmp = (1-c*x)
        tmp2 = tmp**(limit)
        return (1.0-tmp2)/(1+tmp2)

    def _ppf(self, q, c):
        return 1.0/c*(1-((1.0-q)/(1.0+q))**c)

    def _lower(self, c):
        return 0.0

    def _upper(self, c):
        return 1/numpy.where(c < 10**-10, 10**-10, c)


class GeneralizedHalfLogistic(ShiftScaleDistribution):
    """
    Generalized half-logistic distribution

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.GeneralizedHalfLogistic(0.5)
        >>> distribution
        GeneralizedHalfLogistic(0.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.367, 0.691, 1.   , 1.333, 2.   ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.5  , 0.588, 0.642, 0.64 , 0.54 , 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.085, 0.218, 1.681, 0.818])

    """

    def __init__(self, shape, scale=1, shift=0):
        super(GeneralizedHalfLogistic, self).__init__(
            dist=generalized_half_logistic(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
