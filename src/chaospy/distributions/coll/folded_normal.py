"""Folded normal distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class folded_normal(Dist):
    """Folded normal distribution."""

    def __init__(self, c=1):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        return numpy.sqrt(2.0/numpy.pi)*numpy.cosh(c*x)*numpy.exp(-(x*x+c*c)/2.0)

    def _cdf(self, x, c):
        return special.ndtr(x-c) + special.ndtr(x+c) - 1.0

    def _bnd(self, c):
        return 0, 7.5+c


class FoldedNormal(Add):
    """
    Folded normal distribution.

    Args:
        mu (float, Dist): Location parameter in normal distribution
        sigma (float, Dist): Scaling parameter (in both normal and fold)
        loc (float, Dist): Location of fold

    Examples:
        >>> distribution = chaospy.FoldedNormal(3, 2, 1)
        >>> print(distribution)
        FoldedNormal(loc=1, mu=3, sigma=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [3.3224 4.4938 5.5067 6.6832]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.1417 0.1934 0.1932 0.14  ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [5.7901 2.6245 8.2951 4.9109]
        >>> print(numpy.around(distribution.mom(1), 4))
        10.5
    """

    def __init__(self, mu=0, sigma=1, loc=0):
        self._repr = {"mu": mu, "sigma": sigma, "loc": loc}
        Add.__init__(self, left= folded_normal(mu-loc)*sigma, right=loc)
