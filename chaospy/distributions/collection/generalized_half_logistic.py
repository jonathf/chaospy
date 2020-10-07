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
        >>> distribution = chaospy.GeneralizedHalfLogistic(1, 2, 2)
        >>> distribution
        GeneralizedHalfLogistic(1, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.6667, 3.1429, 3.5   , 3.7778])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.36, 0.49, 0.64, 0.81])
        >>> distribution.sample(4).round(4)
        array([3.581 , 2.4126, 3.949 , 3.3013])
        >>> distribution.mom(1).round(4)
        3.2274
    """

    def __init__(self, shape, scale, shift):
        super(GeneralizedHalfLogistic, self).__init__(
            dist=generalized_half_logistic(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
