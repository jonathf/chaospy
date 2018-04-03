"""Exponential Weibull distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add


class exponential_weibull(Dist):
    """Exponential Weibull distribution."""

    def __init__(self, a=1, c=1):
        Dist.__init__(self, a=a, c=c)

    def _pdf(self, x, a, c):
        exc = numpy.exp(-x**c)
        return a*c*(1-exc)**(a-1) * exc * x**(c-1)

    def _cdf(self, x, a, c):
        exm1c = -numpy.expm1(-x**c)
        return (exm1c)**a

    def _ppf(self, q, a, c):
        return (-numpy.log1p(-q**(1.0/a)))**(1.0/c)

    def _bnd(self, x, a, c):
        return 0, self._ppf(1-1e-10, a, c)


class ExponentialWeibull(Add):
    """
    Exponential Weibull distribution.

    Args:
        alpha (float, Dist): First shape parameter
        kappa (float, Dist): Second shape parameter
        scale (float, Dist): Scaling parameter
        shift (float, Dist): Location parameter

    Examples:
        >>> distribution = chaospy.ExponentialWeibull(2, 2, 2, 1)
        >>> print(distribution)
        ExponentialWeibull(alpha=2, kappa=2, scale=2, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [2.5398 3.0009 3.4412 3.9989]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.3807 0.4651 0.4262 0.2832]
        >>> print(numpy.around(distribution.sample(4), 4))
        [3.5711 2.2872 4.8376 3.1776]
        >>> print(numpy.around(distribution.mom(1), 4))
        5.8702
    """
    def __init__(self, alpha=1, kappa=1, scale=1, shift=0):
        self._repr = {
            "alpha": alpha, "kappa": kappa, "scale": scale, "shift": shift}
        Add.__init__(
            self, left=exponential_weibull(alpha, kappa)*scale, right=shift)
