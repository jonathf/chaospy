"""Generalized gamma distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class generalized_gamma(SimpleDistribution):
    """Generalized gamma distribution."""

    def __init__(self, a, c):
        super(generalized_gamma, self).__init__(dict(a=a, c=c))

    def _pdf(self, x, a, c):
        return abs(c)*numpy.exp(
            (c*a-1)*numpy.log(numpy.clip(x, 1e-15, None))-x**c-special.gammaln(a))

    def _cdf(self, x, a, c):
        val = special.gammainc(a, x**c)
        cond = c+0*val
        return numpy.where(cond > 0, val, 1-val)

    def _ppf(self, q, a, c):
        val = numpy.where(c > 0, q, 1-q)
        return special.gammaincinv(a, val)**(1./c)

    def _lower(self, a, c):
        return 0.

    def _upper(self, a, c):
        cond = c > 0
        val = numpy.where(cond, 1-1e-15, 1e-15)
        return special.gammaincinv(a, val)**(1./c)

    def _mom(self, k, a, c):
        return special.gamma((c+k)*1./a)/special.gamma(c*1./a)


class GeneralizedGamma(ShiftScaleDistribution):
    """
    Generalized gamma distribution

    Args:
        shape1 (float, Distribution):
            Shape parameter 1
        shape2 (float, Distribution):
            Shape parameter 2
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.GeneralizedGamma(1.5, 15)
        >>> distribution
        GeneralizedGamma(1.5, 15)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.955, 0.995, 1.026, 1.058, 1.271])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 3.82 , 6.033, 6.76 , 5.555, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.034, 0.928, 1.095, 1.009])
        >>> distribution.mom(1).round(3)
        4.591

    """

    def __init__(self, shape1, shape2, scale=1, shift=0):
        super(GeneralizedGamma, self).__init__(
            dist=generalized_gamma(shape1, shape2),
            scale=scale,
            shift=shift,
            repr_args=[shape1, shape2],
        )
