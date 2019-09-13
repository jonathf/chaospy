"""Arc-Cosines Hyperbolic."""
import numpy

from ..baseclass import Dist
from .. import evaluation, approximation


class Arccosh(Dist):
    """
    Arc-Cosine Hyperbolic.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Arccosh(chaospy.Uniform(1, 2))
        >>> print(distribution)
        Arccosh(Uniform(lower=1, upper=2))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(distribution.inv(q).round(4))
        [0.6224 0.867  1.047  1.1929]
        >>> print(distribution.fwd(distribution.inv(q)).round(4))
        [0.2 0.4 0.6 0.8]
        >>> print(distribution.pdf(distribution.inv(q)).round(4))
        [0.6633 0.9798 1.249  1.4967]
        >>> print(distribution.sample(4).round(4))
        [1.0887 0.4751 1.2878 0.9463]
        >>> print(distribution.mom(1).round(4))
        0.9019
        >>> print(distribution.ttr([0, 1, 2]).round(4))
        [[0.9019 0.7012 0.6772]
         [1.     0.0933 0.1037]]
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        assert numpy.all(dist.range() >= 1)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        return evaluation.evaluate_density(
            dist, numpy.cosh(x), cache=cache)*numpy.sinh(x)

    def _cdf(self, x, dist, cache):
        return evaluation.evaluate_forward(dist, numpy.cosh(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.arccosh(evaluation.evaluate_inverse(dist, q, cache=cache))

    def _bnd(self, x, dist, cache):
        return numpy.arccosh(evaluation.evaluate_bound(
            dist, numpy.cosh(x), cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"

    def _fwd_cache(self, cache):
        dist = evaluation.get_forward_cache(self.prm["dist"], cache)
        if not isinstance(dist, Dist):
            return numpy.arccosh(dist)
        return self

    def _inv_cache(self, cache):
        dist = evaluation.get_forward_cache(self.prm["dist"], cache)
        if not isinstance(dist, Dist):
            return numpy.cosh(dist)
        return self
