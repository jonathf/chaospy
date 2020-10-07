"""Normal (Gaussian) probability distribution."""
import numpy
from scipy import special

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class normal(SimpleDistribution):
    """Standard normal distribution."""

    def __init__(self):
        super(normal, self).__init__()

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
        >>> distribution = chaospy.Normal(2, 2)
        >>> distribution
        Normal(mu=2, sigma=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([0.3168, 1.4933, 2.5067, 3.6832])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.14  , 0.1932, 0.1932, 0.14  ])
        >>> distribution.sample(4).round(4)
        array([ 2.7901, -0.4006,  5.2952,  1.9107])
        >>> distribution.mom(1).round(4)
        2.0
        >>> distribution.ttr([1, 2, 3]).round(4)
        array([[ 2.,  2.,  2.],
               [ 4.,  8., 12.]])
    """

    def __init__(self, mu=0, sigma=1):
        super(Normal, self).__init__(
            dist=normal(), scale=sigma, shift=mu,
        )
        self._repr_args = ["mu=%s" % mu, "sigma=%s" % sigma]
