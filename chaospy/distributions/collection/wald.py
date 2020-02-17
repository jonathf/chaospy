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
        out = numpy.zeros(x.shape)
        indices = x > 0
        out[indices] = 1.0/numpy.sqrt(2*numpy.pi*x[indices])
        out[indices] *= numpy.exp(-(1-mu*x[indices])**2.0 / (2*x[indices]*mu**2.0))
        return out

    def _cdf(self, x, mu):
        trm1 = 1./mu - x
        trm2 = 1./mu + x
        isqx = numpy.tile(numpy.inf, x.shape)
        indices = x > 0
        isqx[indices] = 1./numpy.sqrt(x[indices])
        out = 1.-special.ndtr(isqx*trm1)
        out -= numpy.exp(2.0/mu)*special.ndtr(-isqx*trm2)
        return out

    def _lower(self, mu):
        return 0.

    def _upper(self, mu):
        return 1e10


class Wald(Add):
    """
    Wald distribution.

    Reciprocal inverse Gaussian distribution.

    Args:
        mu (float, Dist):
            Mean of the normal distribution
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.Wald(2, 2, 2)
        >>> distribution
        Wald(mu=2, scale=2, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([2.7154, 3.4499, 4.5777, 6.6903])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.3242, 0.2262, 0.138 , 0.063 ])
        >>> distribution.sample(4).round(4)
        array([ 9.7718,  2.3482, 10.1276,  5.6088])
    """

    def __init__(self, mu=1, scale=1, shift=0):
        self._repr = {"mu": mu, "scale": scale, "shift": shift}
        Add.__init__(self, left=wald(mu)*scale, right=shift)
