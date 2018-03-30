"""Wald distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class wald(Dist):
    """Wald distribution."""

    def __init__(self, mu):
        Dist.__init__(self, mu=mu)

    def _pdf(self, x, mu):
        return 1.0/numpy.sqrt(2*numpy.pi*x)*numpy.exp(-(1-mu*x)**2.0 / (2*x*mu**2.0))

    def _cdf(self, x, mu):
        trm1 = 1./mu - x
        trm2 = 1./mu + x
        isqx = 1./numpy.sqrt(x)
        return 1.-special.ndtr(isqx*trm1)-numpy.exp(2.0/mu)*special.ndtr(-isqx*trm2)

    def _bnd(self, mu):
        return 0.0, 10**10


class Wald(Add):
    """
    Wald distribution.

    Reciprocal inverse Gaussian distribution.

    Args:
        mu (float, Dist) : Mean of the normal distribution
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.Wald(2, 2, 2)
        >>> print(distribution)
        Wald(mu=2, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [2.7154 3.45   4.5777 6.6903]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.3242 0.2262 0.138  0.063 ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 4.9997  2.4662 11.33    3.848 ]
    """

    def __init__(self, mu=0, scale=1, shift=0):
        self._repr = {"mu": mu, "scale": scale, "shift": shift}
        Add.__init__(self, left=wald(mu)*scale, right=shift)
