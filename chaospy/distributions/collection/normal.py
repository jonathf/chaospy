"""Normal (Gaussian) probability distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class normal(Dist):
    """Standard normal distribution."""

    def __init__(self):
        Dist.__init__(self)

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

    def _lower(self):
        return -7.5

    def _upper(self):
        return 7.5


class Normal(Add):
    R"""
    Normal (Gaussian) distribution

    Args:
        mu (float, Dist):
            Mean of the distribution.
        sigma (float, Dist):
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
        self._repr = {"mu": mu, "sigma": sigma}
        Add.__init__(self, left=normal()*sigma, right=mu)
