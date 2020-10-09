"""Exponential power distribution."""
import numpy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class exponential_power(SimpleDistribution):
    """Exponential power distribution."""

    def __init__(self, b=1):
        super(exponential_power, self).__init__(dict(b=b))

    def _pdf(self, x, b):
        xbm1 = x**(b-1.0)
        xb = xbm1*x
        return numpy.exp(1)*b*xbm1*numpy.exp(xb-numpy.exp(xb))

    def _cdf(self, x, b):
        return -numpy.expm1(-numpy.expm1(x**b))

    def _ppf(self, q, b):
        return pow(numpy.log1p(-numpy.log1p(-q)), 1./b)

    def _lower(self, b):
        del b
        return 0.

    def _upper(self, b):
        return 3.6**(1./b)


class ExponentialPower(ShiftScaleDistribution):
    """
    Exponential power distribution.

    Also known as Generalized error distribution and Generalized normal
    distribution version 1.

    Args:
        shape (float, Distribution):
            Shape parameter
        scale (float, Distribution):
            Scaling parameter
        shift (float, Distribution):
            Location parameter

    Examples:
        >>> distribution = chaospy.ExponentialPower(1.5)
        >>> distribution
        ExponentialPower(1.5)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([0.   , 0.344, 0.554, 0.751, 0.973, 2.349])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.86 , 1.012, 0.996, 0.772, 0.   ])
        >>> distribution.sample(4).round(3)
        array([0.805, 0.237, 1.243, 0.635])

    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(ExponentialPower, self).__init__(
            dist=exponential_power(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
