"""Exponential Weibull distribution."""
import numpy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class exponential_weibull(SimpleDistribution):
    """Exponential Weibull distribution."""

    def __init__(self, a=1, c=1):
        super(exponential_weibull, self).__init__(dict(a=a, c=c))

    def _pdf(self, x, a, c):
        exc = numpy.exp(-x**c)
        return a*c*(1-exc)**(a-1)*exc*x**(c-1)

    def _cdf(self, x, a, c):
        exm1c = -numpy.expm1(-x**c)
        return (exm1c)**a

    def _ppf(self, q, a, c):
        return (-numpy.log1p(-q**(1./a)))**(1./c)

    def _lower(self, a, c):
        return 0.

    def _upper(self, a, c):
        return (-numpy.log1p(-(1-1e-15)**(1./a)))**(1./c)


class ExponentialWeibull(ShiftScaleDistribution):
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
        >>> distribution = chaospy.ExponentialWeibull(alpha=2, kappa=3)
        >>> distribution
        ExponentialWeibull(2, 3)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.84 , 1.   , 1.142, 1.31 , 3.275])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 1.047, 1.396, 1.367, 0.972, 0.   ])
        >>> distribution.sample(4).round(3)
        array([1.182, 0.745, 1.544, 1.058])

    """
    def __init__(self, alpha=1, kappa=1, scale=1, shift=0):
        super(ExponentialWeibull, self).__init__(
            dist=exponential_weibull(alpha, kappa),
            scale=scale,
            shift=shift,
            repr_args=[alpha, kappa],
        )
