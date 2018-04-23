"""Tangent Hyperbolic."""
import numpy

from ..baseclass import Dist
from .. import evaluation, approximation


class Tanh(Dist):
    """
    Tangent Hyperbolic.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Tanh(chaospy.Uniform(0, 1))
        >>> print(distribution)
        Tanh(Uniform(lower=0, upper=1))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.1974 0.3799 0.537  0.664 ]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.0201 1.0811 1.1855 1.3374]
        >>> print(numpy.around(distribution.sample(4), 4))
        [0.5741 0.1145 0.7399 0.448 ]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.3808
        >>> print(numpy.around(distribution.ttr([0, 1, 2]), 4))
        [[0.3519 0.4807 0.2854]
         [1.     0.0456 0.0385]]
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        assert numpy.all(dist.range() <= 1)
        assert numpy.all(dist.range() >= -1)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, numpy.arctanh(x), cache=cache)/numpy.sqrt(1-x*x)

    def _cdf(self, x, dist, cache):
        return evaluation.evaluate_forward(dist, numpy.arctanh(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.tanh(evaluation.evaluate_inverse(dist, q, cache=cache))

    def _bnd(self, x, dist, cache):
        return numpy.tanh(evaluation.evaluate_bound(
            dist, numpy.arctanh(x), cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"
