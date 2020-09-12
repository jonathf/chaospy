"""Burr Type III distribution."""
import numpy
from scipy import special

from ..baseclass import DistributionCore, ShiftScale


class burr(DistributionCore):
    """Stadard Burr distribution."""

    def __init__(self, alpha=1., kappa=1.):
        super(burr, self).__init__(alpha=alpha, kappa=kappa)

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


class Burr(ShiftScale):
    """
    Burr Type XII or Singh-Maddala distribution.

    Args:
        alpha (float, Distribution):
            First shape parameter
        kappa (float, Distribution):
            Second shape parameter
        loc (float, Distribution):
            Location parameter
        scale (float, Distribution):
            Scaling parameter

    Examples:
        >>> distribution = chaospy.Burr(100, 1.2, 2, 4)
        >>> distribution
        Burr(100, 1.2, scale=2, shift=4)
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

    def __init__(self, alpha=1, kappa=1, scale=1, shift=0):
        super(Burr, self).__init__(
            dist=burr(alpha, kappa),
            scale=scale,
            shift=shift,
            repr_args=[alpha, kappa],
        )
