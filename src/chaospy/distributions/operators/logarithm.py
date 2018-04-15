"""Logarithm with base Euler's constant."""
import numpy

from ..baseclass import Dist
from .. import evaluation, approximation


class Log(Dist):
    """
    Logarithm with base Euler's constant.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Log(chaospy.Uniform(1, 2))
        >>> print(distribution)
        Log(Uniform(lower=1, upper=2))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.1823 0.3365 0.47   0.5878]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.2 1.4 1.6 1.8]
        >>> print(numpy.around(distribution.sample(4), 4))
        [0.5029 0.1089 0.668  0.3935]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.3466
        >>> print(numpy.around(distribution.ttr([0, 1, 2]), 4))
        [[0.3863 0.3389 0.3454]
         [1.     0.0391 0.0321]]
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        assert numpy.all(dist.range() > 0)
        Dist.__init__(self, dist=dist)

    def _pdf(self, xloc, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, numpy.e**xloc, cache=cache)*numpy.e**xloc

    def _cdf(self, xloc, dist, cache):
        """Cumulative distribution function."""
        return evaluation.evaluate_forward(dist, numpy.e**xloc, cache=cache)

    def _ppf(self, q, dist, cache):
        """Point percentile function."""
        return numpy.log(evaluation.evaluate_inverse(dist, q, cache=cache))

    def _bnd(self, xloc, dist, cache):
        """Distribution bounds."""
        return numpy.log(evaluation.evaluate_bound(
            dist, numpy.e**xloc, cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"
