"""Arc-Cosine."""
import numpy

from ..baseclass import Dist
from .. import evaluation, approximation


class Arccos(Dist):
    """
    Arc-Cosine.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Arccos(chaospy.Uniform(0, 1))
        >>> print(distribution)
        Arccos(Uniform(lower=0, upper=1))
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.6435 0.9273 1.1593 1.3694]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.6    0.8    0.9165 0.9798]
        >>> print(numpy.around(distribution.sample(4), 4))
        [1.2171 0.4843 1.5211 1.0265]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.7854
        >>> print(numpy.around(distribution.ttr([0, 1, 2]), 4))
        [[1.     0.8406 0.8083]
         [1.     0.1416 0.1492]]
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        assert numpy.all(dist.range() >= -1)
        assert numpy.all(dist.range() <= 1)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        return evaluation.evaluate_density(
            dist, numpy.cos(x), cache=cache)*numpy.sin(x)

    def _cdf(self, x, dist, cache):
        return 1-evaluation.evaluate_forward(dist, numpy.cos(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.arccos(evaluation.evaluate_inverse(dist, 1-q, cache=cache))

    def _bnd(self, x, dist, cache):
        return numpy.arccos(evaluation.evaluate_bound(
            dist, numpy.cos(x), cache=cache))[::-1]

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"
