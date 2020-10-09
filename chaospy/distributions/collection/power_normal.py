"""Power normal or Box-Cox distribution."""
import numpy
from scipy import special
import chaospy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class power_normal(SimpleDistribution):
    """Power normal or Box-Cox distribution."""

    def __init__(self, c):
        super(power_normal, self).__init__(dict(c=c))

    def _pdf(self, x, c):
        norm = (2*numpy.pi)**-.5*numpy.exp(-x**2/2.)
        return c*norm*special.ndtr(-x)**(c-1.)

    def _cdf(self, x, c):
        return 1.-special.ndtr(-x)**c

    def _ppf(self, q, c):
        return -special.ndtri(pow(1-q, 1./c))

    def _lower(self, c):
        return -special.ndtri(pow(1-1e-15, 1./c))

    def _upper(self, c):
        return -special.ndtri(pow(1e-15, 1./c))


class PowerNormal(ShiftScaleDistribution):
    """
    Power normal or Box-Cox distribution.

    Args:
        shape (float, Distribution):
            Shape parameter
        mu (float, Distribution):
            Mean of the normal distribution
        scale (float, Distribution):
            Standard deviation of the normal distribution

    Examples:
        >>> distribution = chaospy.PowerNormal(1)
        >>> distribution
        PowerNormal(1)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-7.941, -0.842, -0.253,  0.253,  0.842,  7.941])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.28 , 0.386, 0.386, 0.28 , 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 0.395, -1.2  ,  1.648, -0.045])

    """

    def __init__(self, shape=1, mu=0, sigma=1):
        super(PowerNormal, self).__init__(
            dist=power_normal(shape),
            scale=sigma,
            shift=mu,
        )
        self._repr_args = [shape]
        self._repr_args += chaospy.format_repr_kwargs(mu=(mu, 0))
        self._repr_args += chaospy.format_repr_kwargs(sigma=(sigma, 1))
