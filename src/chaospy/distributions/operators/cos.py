"""Cosine."""
import numpy

from ..baseclass import Dist
from .. import evaluation, approximation


class Cos(Dist):
    """
    Cosine.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Cos(chaospy.Uniform(0, 1))
        >>> print(distribution)
        Cos(Uniform(lower=0, upper=1))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.6967 0.8253 0.9211 0.9801]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.394  1.771  2.5679 5.0335]
        >>> print(numpy.around(distribution.sample(4), 4))
        [0.9406 0.6333 0.9988 0.8689]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.7702
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, numpy.arccos(x), cache=cache)/numpy.sqrt(1-x*x)

    def _cdf(self, x, dist, cache):
        return 1-evaluation.evaluate_forward(dist, numpy.arccos(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.cos(evaluation.evaluate_inverse(dist, 1-q, cache=cache))

    def _bnd(self, x, dist, cache):
        out = numpy.cos(evaluation.evaluate_bound(
            dist, numpy.arccos(x), cache=cache))
        return out[::-1]

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"

