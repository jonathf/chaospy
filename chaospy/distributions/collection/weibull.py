"""Weibull Distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class weibull(Dist):
    """Weibull Distribution."""

    def __init__(self, a=1):
        Dist.__init__(self, a=a)

    def _pdf(self, x, a):
        return a*x**(a-1)*numpy.e**(-x**a)

    def _cdf(self, x, a):
        return (1-numpy.e**(-x**a))

    def _ppf(self, q, a):
        return (-numpy.log(1-q+1*(q==1)))**(1./a)*(q!=1) + 30.**(1./a)*(q==1)

    def _mom(self, k, a):
        return special.gamma(1.+k*1./a)

    def _lower(self, a):
        return 0.

    def _upper(self, a):
        return 30.**(1./a)


class Weibull(Add):
    """
    Weibull Distribution

    Args:
        shape (float, Dist):
            Shape parameter.
        scale (float, Dist):
            Scale parameter.
        shift (float, Dist):
            Location of lower bound.

    Examples:
        >>> distribution = chaospy.Weibull(2)
        >>> distribution
        Weibull(scale=1, shape=2, shift=0)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([0.4724, 0.7147, 0.9572, 1.2686])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.7558, 0.8577, 0.7658, 0.5075])
        >>> distribution.sample(4).round(4)
        array([1.0296, 0.3495, 1.7325, 0.8113])
        >>> distribution.mom(1).round(4)
        0.8862
    """

    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=weibull(shape)*scale, right=shift)
