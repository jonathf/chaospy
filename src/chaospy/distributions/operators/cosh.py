"""Cosines Hyperbolic."""
import numpy

from ..baseclass import Dist
from .. import evaluation, approximation


class Cosh(Dist):
    """
    Cosines Hyperbolic.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Cosh(chaospy.Uniform(0, 1))
        >>> print(distribution)
        Cosh(Uniform(lower=0, upper=1))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [1.0201 1.0811 1.1855 1.3374]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [4.9668 2.4346 1.5707 1.126 ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [1.2213 1.0066 1.4865 1.1185]
        >>> print(numpy.around(distribution.mom(1), 4))
        1.2715
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, numpy.arccosh(x), cache=cache)/numpy.sqrt(x*x-1)

    def _cdf(self, x, dist, cache):
        return evaluation.evaluate_forward(dist, numpy.arccosh(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.cosh(evaluation.evaluate_inverse(dist, q, cache=cache))

    def _bnd(self, x, dist, cache):
        return numpy.cosh(evaluation.evaluate_bound(
            dist, numpy.arccosh(x), cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"
