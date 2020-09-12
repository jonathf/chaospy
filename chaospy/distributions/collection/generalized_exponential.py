"""Generalized exponential distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class generalized_exponential(DistributionCore):
    """Generalized exponential distribution."""

    def __init__(self, a=1, b=1, c=1):
        super(generalized_exponential, self).__init__(a=a, b=b, c=c)

    def _pdf(self, x, a, b, c):
        return (a+b*(-numpy.expm1(-c*x)))*numpy.exp((-a-b)*x+b*(-numpy.expm1(-c*x))/c)

    def _cdf(self, x, a, b, c):
        output = -numpy.expm1((-a-b)*x + b*(-numpy.expm1(-c*x))/c)
        output = numpy.where(x > 0, output, 0)
        return output

    def _lower(self, a, b, c):
        return 0.

    def _upper(self, a, b, c):
        return 10**10


class GeneralizedExponential(ShiftScale):
    """
    Generalized exponential distribution.

    Args:
        a (float, Distribution):
            First shape parameter
        b (float, Distribution):
            Second shape parameter
        c (float, Distribution):
            Third shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Note:
        "An Extension of Marshall and Olkin's Bivariate Exponential Distribution",
        H.K. Ryu, Journal of the American Statistical Association, 1993.

        "The Exponential Distribution: Theory, Methods and Applications",
        N. Balakrishnan, Asit P. Basu.

    Examples:
        >>> distribution = chaospy.GeneralizedExponential(3, 2, 2, 2, 2)
        >>> distribution
        GeneralizedExponential(3, 2, 2, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.1423, 2.3113, 2.5314, 2.8774])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([1.3061, 1.0605, 0.7649, 0.4168])
        >>> distribution.sample(4).round(4)
        array([3.3106, 2.0498, 3.3575, 2.7079])
    """

    def __init__(self, a=1, b=1, c=1, scale=1, shift=0):
        super(GeneralizedExponential, self).__init__(
            dist=generalized_exponential(a, b, c),
            scale=scale,
            shift=shift,
            repr_args=[a, b, c],
        )
