"""Power log-Normal probability distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class power_log_normal(Dist):
    """Power log-Normal probability distribution."""

    def __init__(self, c, s):
        Dist.__init__(self, c=c, s=s)

    def _pdf(self, x, c, s):
        norm = (2*numpy.pi)**-.5*numpy.exp(-(numpy.log(x)/s)**2/2.)
        return c/(x*s)*norm*pow(special.ndtr(-numpy.log(x)/s), c*1.-1.)

    def _cdf(self, x, c, s):
        return 1. - pow(special.ndtr(-numpy.log(x)/s), c*1.)

    def _ppf(self, q, c, s):
        return numpy.exp(-s*special.ndtri(pow(1.-q, 1./c)))

    def _bnd(self, x, c, s):
        return 0., self._ppf(1-1e-10, c, s)


class PowerLogNormal(Add):
    """
    Power log-normal distribution

    Args:
        shape (float, Dist) : Shape parameter
        mu (float, Dist) : Mean in the normal distribution.  Overlaps with
                scale by mu=log(scale)
        sigma (float, Dist) : Standard deviation of the normal distribution.
        shift (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter. Overlap with mu in scale=e**mu

    Examples:
        >>> distribution = chaospy.PowerLogNormal(2, 2, 2, 2, 2)
        >>> print(distribution)
        PowerLogNormal(mu=2, scale=2, shape=2, shift=2, sigma=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [ 3.212   5.2707  9.5114 21.2701]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.1347 0.0711 0.0317 0.0092]
        >>> print(numpy.around(distribution.sample(4), 4))
        [11.4445  2.6512 69.8654  6.6177]
    """
    def __init__(self, shape=1, mu=0, sigma=1, shift=0, scale=1):
        self._repr = {
            "shape": shape, "mu": mu, "sigma": sigma, "shift": shift, "scale": scale}
        Add.__init__(self, left=power_log_normal(
            shape, sigma)*scale*numpy.e**mu, right=shift)
