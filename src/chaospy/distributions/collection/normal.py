"""Normal (Gaussian) probability distribution."""
import numpy
from scipy import special, misc

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
        return .5*misc.factorial2(k-1)*(1+(-1)**k)

    def _ttr(self, n):
        return 0., 1.*n

    def _bnd(self, x):
        return -7.5, 7.5


class Normal(Add):
    R"""
    Normal (Gaussian) distribution

    Args:
        mu (float, Dist) : Mean of the distribution.
        sigma (float, Dist) : Standard deviation.  sigma > 0

    Examples:
        >>> distribution = chaospy.Normal(2, 2)
        >>> print(distribution)
        Normal(mu=2, sigma=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.3168 1.4933 2.5067 3.6832]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.14   0.1932 0.1932 0.14  ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 2.7901 -0.4006  5.2952  1.9107]
        >>> print(numpy.around(distribution.mom(1), 4))
        2.0
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[ 2.  2.  2.]
         [ 4.  8. 12.]]
    """

    def __init__(self, mu=0, sigma=1):
        self._repr = {"mu": mu, "sigma": sigma}
        Add.__init__(self, left=normal()*sigma, right=mu)
