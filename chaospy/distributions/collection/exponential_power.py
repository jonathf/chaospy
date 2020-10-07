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
        >>> distribution = chaospy.ExponentialPower(2, 2, 1)
        >>> distribution
        ExponentialPower(2, scale=2, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([1.8976, 2.2848, 2.6129, 2.9587])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.4392, 0.5823, 0.6182, 0.5111])
        >>> distribution.sample(4).round(4)
        array([2.7003, 1.679 , 3.3551, 2.4223])
        >>> distribution.mom(1).round(4)
        2.4314
    """

    def __init__(self, shape=1, scale=1, shift=0):
        super(ExponentialPower, self).__init__(
            dist=exponential_power(shape),
            scale=scale,
            shift=shift,
            repr_args=[shape],
        )
