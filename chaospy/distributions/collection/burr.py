"""Burr Type III distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class burr(SimpleDistribution):
    """Stadard Burr distribution."""

    def __init__(self, alpha=1., kappa=1.):
        super(burr, self).__init__(dict(alpha=alpha, kappa=kappa))

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
        del alpha
        del kappa
        return 0.

    def _upper(self, alpha, kappa):
        return (1e12**(1./kappa)-1)**(1./alpha)


class Burr(ShiftScaleDistribution):
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
        >>> distribution = chaospy.Burr(5, 2)
        >>> distribution
        Burr(5, 2)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([ 0.   ,  0.652,  0.781,  0.897,  1.043, 15.849])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([ 0.   , 92.945, 20.444,  4.852,  0.694,  0.   ])
        >>> distribution.sample(4).round(3)
        array([0.931, 0.575, 1.284, 0.828])
        >>> distribution.mom(1).round(3)
        1.283

    """

    def __init__(self, alpha=1, kappa=1, scale=1, shift=0):
        super(Burr, self).__init__(
            dist=burr(alpha, kappa),
            scale=scale,
            shift=shift,
            repr_args=[alpha, kappa],
        )
