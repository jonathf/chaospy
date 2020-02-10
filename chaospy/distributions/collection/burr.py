"""Burr Type III distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class burr(Dist):
    """Stadard Burr distribution."""

    def __init__(self, alpha=1., kappa=1.):
        Dist.__init__(self, alpha=alpha, kappa=kappa)

    def _pdf(self, x, alpha, kappa):
        output = numpy.zeros(x.shape)
        indices = x > 0
        output[indices] = alpha*kappa*x[indices]**(-alpha-1.0)
        output[indices] /= (1+x[indices]**alpha)**(kappa+1)
        return output

    def _cdf(self, x, alpha, kappa):
        output = numpy.zeros(x.shape)
        indices = x > 0
        output[indices] = 1-(1+x[indices]**alpha)**-kappa
        return output

    def _ppf(self, q, alpha, kappa):
        return ((1-q)**(-1./kappa) -1)**(1./alpha)

    def _mom(self, k, alpha, kappa):
        return kappa*special.beta(1-k*1./alpha, kappa+k*1./alpha)

    def _lower(self, alpha, kappa):
        return 0.


class Burr(Add):
    """
    Burr Type XII or Singh-Maddala distribution.

    Args:
        alpha (float, Dist):
            First shape parameter
        kappa (float, Dist):
            Second shape parameter
        loc (float, Dist):
            Location parameter
        scale (float, Dist):
            Scaling parameter

    Examples:
        >>> distribution = chaospy.Burr(100, 1.2, 4, 2)
        >>> distribution
        Burr(alpha=100, kappa=1.2, loc=4, scale=2)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> distribution.inv(q).round(4)
        array([5.9642, 5.9819, 5.9951, 6.0081, 6.0249])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.1667, 0.3333, 0.5   , 0.6667, 0.8333])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([266.5437,  71.6255,  21.5893,   5.3229,   0.643 ])
        >>> distribution.sample(4).round(4)
        array([6.007 , 5.9558, 6.0489, 5.9937])
        >>> distribution.mom(1).round(4)
        6.0061
    """

    def __init__(self, alpha=1, kappa=1, loc=0, scale=1):
        self._repr = {
            "alpha": alpha, "kappa": kappa, "loc": loc, "scale": scale}
        Add.__init__(self, left=burr(alpha, kappa)*scale, right=loc)
