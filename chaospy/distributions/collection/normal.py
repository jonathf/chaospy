"""Normal (Gaussian) probability distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class normal(SimpleDistribution):
    """Standard normal distribution."""

    def __init__(self):
        super(normal, self).__init__()

    def _lower(self):
        return -8.22

    def _upper(self):
        return 8.22

    def _pdf(self, x):
        return (2*numpy.pi)**(-.5)*numpy.e**(-x**2/2.)

    def _cdf(self, x):
        return special.ndtr(x)

    def _ppf(self, x):
        return special.ndtri(x)

    def _mom(self, k):
        return .5*special.factorial2(k-1)*(1+(-1)**k)

    def _ttr(self, n):
        return 0., 1.*n


class Normal(ShiftScaleDistribution):
    R"""
    Normal (Gaussian) distribution

    Args:
        mu (float, Distribution):
            Mean of the distribution.
        sigma (float, Distribution):
            Standard deviation.  sigma > 0

    Examples:
        >>> distribution = chaospy.Normal(2, 3)
        >>> distribution
        Normal(mu=2, sigma=3)
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-22.66 ,  -0.525,   1.24 ,   2.76 ,   4.525,  26.66 ])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.   , 0.093, 0.129, 0.129, 0.093, 0.   ])
        >>> distribution.sample(4).round(3)
        array([ 3.185, -1.601,  6.943,  1.866])
        >>> distribution.mom(1).round(3)
        2.0
        >>> distribution.ttr([0, 1, 2, 3]).round(3)
        array([[ 2.,  2.,  2.,  2.],
               [ 0.,  9., 18., 27.]])

    """

    def __init__(self, mu=0, sigma=1):
        super(Normal, self).__init__(dist=normal(), scale=sigma, shift=mu)
        self._repr_args = ["mu=%s" % mu, "sigma=%s" % sigma]
