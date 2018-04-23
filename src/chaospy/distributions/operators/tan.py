"""Tangent."""
import numpy

from ..baseclass import Dist
from .. import evaluation, approximation


class Tan(Dist):
    """
    Tangent.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Tan(chaospy.Uniform(0, 1))
        >>> print(distribution)
        Tan(Uniform(lower=0, upper=1))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.2027 0.4228 0.6841 1.0296]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.9605 0.8484 0.6812 0.4854]
        >>> print(numpy.around(distribution.sample(4), 4))
        [0.7659 0.1155 1.3992 0.5234]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.7787
        >>> print(numpy.around(distribution.ttr([0, 1, 2]), 4))
        [[0.6156 0.8079 0.7829]
         [1.     0.1784 0.1654]]
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, numpy.arctan(x), cache=cache)/(1+x*x)

    def _cdf(self, x, dist, cache):
        return evaluation.evaluate_forward(dist, numpy.arctan(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.tan(evaluation.evaluate_inverse(
            dist, q, cache=cache))

    def _bnd(self, x, dist, cache):
        return numpy.tan(evaluation.evaluate_bound(
            dist, numpy.arctan(x), cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"

