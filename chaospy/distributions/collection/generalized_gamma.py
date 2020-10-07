"""Generalized gamma distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class generalized_gamma(SimpleDistribution):
    """Generalized gamma distribution."""

    def __init__(self, a, c):
        super(generalized_gamma, self).__init__(dict(a=a, c=c))

    def _pdf(self, x, a, c):
        return abs(c)*numpy.exp((c*a-1)*numpy.log(x)-x**c-special.gammaln(a))

    def _cdf(self, x, a, c):
        val = special.gammainc(a, x**c)
        cond = c+0*val
        return numpy.where(cond > 0, val, 1-val)

    def _ppf(self, q, a, c):
        val1 = special.gammaincinv(a, q)
        val2 = special.gammaincinv(a, 1.0-q)
        ic = 1.0/c
        cond = c+0*val1
        return numpy.where(cond > 0, val1**ic, val2**ic)

    def _mom(self, k, a, c):
        return special.gamma((c+k)*1./a)/special.gamma(c*1./a)

    def _lower(self, a, c):
        return 0.


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
        >>> distribution = chaospy.GeneralizedGamma(3, 2, 2, 2)
        >>> distribution
        GeneralizedGamma(3, 2, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([4.4779, 5.0233, 5.5244, 6.1372])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.3145, 0.4016, 0.3807, 0.2624])
        >>> distribution.sample(4).round(4)
        array([5.6691, 4.1674, 7.0214, 5.2264])
        >>> distribution.mom(1).round(4)
        3.477
    """

    def __init__(self, shape1, shape2, scale, shift):
        super(GeneralizedGamma, self).__init__(
            dist=generalized_gamma(shape1, shape2),
            scale=scale,
            shift=shift,
            repr_args=[shape1, shape2],
        )
