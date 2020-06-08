"""Power normal or Box-Cox distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class power_normal(Dist):
    """Power normal or Box-Cox distribution."""

    def __init__(self, c):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        norm = (2*numpy.pi)**-.5*numpy.exp(-x**2/2.)
        return c*norm*special.ndtr(-x)**(c-1.)

    def _cdf(self, x, c):
        return 1.-special.ndtr(-x)**c

    def _ppf(self, q, c):
        return -special.ndtri(pow(1-q, 1./c))


class PowerNormal(Add):
    """
    Power normal or Box-Cox distribution.

    Args:
        shape (float, Dist):
            Shape parameter
        mu (float, Dist):
            Mean of the normal distribution
        scale (float, Dist):
            Standard deviation of the normal distribution

    Examples:
        >>> distribution = chaospy.PowerNormal(2, 2, 2)
        >>> distribution
        PowerNormal(mu=2, scale=2, shape=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([-0.5008,  0.4919,  1.3233,  2.2654])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.1633, 0.2325, 0.2383, 0.1768])
        >>> distribution.sample(4).round(4)
        array([ 1.5523, -1.122 ,  3.5244,  0.8368])
        >>> distribution.mom(1).round(4)
        0.8716
    """

    def __init__(self, shape=1, mu=0, scale=1):
        self._repr = {"shape": shape, "mu": mu, "scale": scale}
        Add.__init__(self, left=power_normal(shape)*scale, right=mu)
