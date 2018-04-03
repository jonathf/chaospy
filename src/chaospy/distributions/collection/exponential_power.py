"""Exponential power distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add


class exponential_power(Dist):
    """Exponential power distribution."""

    def __init__(self, b=1):
        Dist.__init__(self, b=b)

    def _pdf(self, x, b):
        xbm1 = x**(b-1.0)
        xb = xbm1 * x
        return numpy.exp(1)*b*xbm1 * numpy.exp(xb - numpy.exp(xb))

    def _cdf(self, x, b):
        xb = x**b
        return -numpy.expm1(-numpy.expm1(xb))

    def _ppf(self, q, b):
        return pow(numpy.log1p(-numpy.log1p(-q)), 1.0/b)

    def _bnd(self, x, b):
        return 0, self._ppf(1-1e-10, b)


class ExponentialPower(Add):
    """
    Exponential power distribution.

    Also known as Generalized error distribution and Generalized normal
    distribution version 1.

    Args:
        shape (float, Dist): Shape parameter
        scale (float, Dist): Scaling parameter
        shift (float, Dist): Location parameter

    Examples:
        >>> distribution = chaospy.ExponentialPower(2, 2, 1)
        >>> print(distribution)
        ExponentialPower(scale=2, shape=2, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [1.8976 2.2848 2.6129 2.9587]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.4392 0.5823 0.6182 0.5111]
        >>> print(numpy.around(distribution.sample(4), 4))
        [2.7003 1.679  3.3551 2.4223]
        >>> print(numpy.around(distribution.mom(1), 4))
        2.783
    """

    def __init__(self, shape=0, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=exponential_power(shape)*scale, right=shift)
