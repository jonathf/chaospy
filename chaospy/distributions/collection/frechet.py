"""Frechet or Extreme value distribution type 2."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class frechet(Dist):
    """Frechet or Extreme value distribution type 2."""

    def __init__(self, c=1):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        return c*pow(x,c-1)*numpy.exp(-pow(x,c))

    def _cdf(self, x, c):
        return -numpy.expm1(-pow(x,c))

    def _ppf(self, q, c):
        return pow(-numpy.log1p(-q),1.0/c)

    def _mom(self, k, c):
        return special.gamma(1-k*1./c)

    def _lower(self, c):
        return 0.


class Frechet(Add):
    """
    Frechet or Extreme value distribution type 2.

    Args:
        shape (float, Dist):
            Shape parameter
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.Frechet(3, 2, 1)
        >>> distribution
        Frechet(scale=2, shape=3, shift=1)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.2131, 2.5988, 2.9426, 3.3438])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.4415, 0.5751, 0.566 , 0.412 ])
        >>> distribution.sample(4).round(4)
        array([3.0393, 1.9924, 3.8849, 2.7397])
        >>> distribution.mom(1).round(4)
        3.7082
    """

    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=frechet(shape)*scale, right=shift)
