"""Logarithm with base 10."""
import numpy

from ..baseclass import Dist
from .. import evaluation, approximation


class Log10(Dist):
    """
    Logarithm with base 10.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Log10(chaospy.Uniform(1, 2))
        >>> print(distribution)
        Log10(Uniform(lower=1, upper=2))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.0792 0.1461 0.2041 0.2553]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [2.7631 3.2236 3.6841 4.1447]
        >>> print(numpy.around(distribution.sample(4), 4))
        [0.2184 0.0473 0.2901 0.1709]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.1505
        >>> print(numpy.around(distribution.ttr([0, 1, 2]), 4))
        [[0.1678 0.1472 0.15  ]
         [1.     0.0074 0.0061]]
    """

    def __init__(self, dist):
        """
        Constructor.

        Args:
            dist (Dist) : distribution (>=0).
        """
        assert isinstance(dist, Dist)
        assert numpy.all(dist.range() > 0)
        Dist.__init__(self, dist=dist)

    def _pdf(self, xloc, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, 10**xloc, cache=cache)*10**xloc*numpy.log(10)

    def _cdf(self, xloc, dist, cache):
        """Cumulative distribution function."""
        return evaluation.evaluate_forward(dist, 10**xloc, cache=cache)

    def _ppf(self, q, dist, cache):
        """Point percentile function."""
        return numpy.log10(evaluation.evaluate_inverse(dist, q, cache=cache))

    def _bnd(self, xloc, dist, cache):
        """Distribution bounds."""
        return numpy.log10(evaluation.evaluate_bound(
            dist, 10**xloc, cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"
