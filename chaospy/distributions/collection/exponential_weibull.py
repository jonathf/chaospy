"""Exponential Weibull distribution."""
import numpy

from ..baseclass import DistributionCore, ShiftScale


class exponential_weibull(DistributionCore):
    """Exponential Weibull distribution."""

    def __init__(self, a=1, c=1):
        super(exponential_weibull, self).__init__(a=a, c=c)

    def _pdf(self, x, a, c):
        exc = numpy.exp(-x**c)
        return a*c*(1-exc)**(a-1) * exc * x**(c-1)

    def _cdf(self, x, a, c):
        exm1c = -numpy.expm1(-x**c)
        return (exm1c)**a

    def _ppf(self, q, a, c):
        return (-numpy.log1p(-q**(1.0/a)))**(1.0/c)

    def _lower(self, a, c):
        return 0.


class ExponentialWeibull(ShiftScale):
    """
    Exponential Weibull distribution.

    Args:
        alpha (float, Distribution):
            First shape parameter
        kappa (float, Distribution):
            Second shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.ExponentialWeibull(2, 2, 2, 1)
        >>> distribution
        ExponentialWeibull(2, 2, scale=2, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.5398, 3.0009, 3.4412, 3.9989])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.3807, 0.4651, 0.4262, 0.2832])
        >>> distribution.sample(4).round(4)
        array([3.5711, 2.2872, 4.8376, 3.1776])
        >>> distribution.mom(1).round(4)
        3.2916
    """
    def __init__(self, alpha=1, kappa=1, scale=1, shift=0):
        super(ExponentialWeibull, self).__init__(
            dist=exponential_weibull(alpha, kappa),
            scale=scale,
            shift=shift,
            repr_args=[alpha, kappa],
        )
