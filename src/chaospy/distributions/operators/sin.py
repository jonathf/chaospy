"""Sinus."""
import numpy
from ..baseclass import Dist
from .. import evaluation, approximation


class Sin(Dist):
    """
    Sinus.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Sin(chaospy.Uniform(0, 1))
        >>> print(distribution)
        Sin(Uniform(lower=0, upper=1))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.1987 0.3894 0.5646 0.7174]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.0203 1.0857 1.2116 1.4353]
        >>> print(numpy.around(distribution.sample(4), 4))
        [0.608  0.1148 0.8136 0.4637]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.4207
        >>> print(numpy.around(distribution.ttr([0, 1, 2]), 4))
        [[0.4597 0.4138 0.4192]
         [1.     0.0614 0.0466]]
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, numpy.arcsin(x), cache=cache)/numpy.sqrt(1-x*x)

    def _cdf(self, x, dist, cache):
        return evaluation.evaluate_forward(dist, numpy.arcsin(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.sin(evaluation.evaluate_inverse(dist, q, cache=cache))

    def _bnd(self, x, dist, cache):
        return numpy.sin(evaluation.evaluate_bound(
            dist, numpy.arcsin(x), cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"

