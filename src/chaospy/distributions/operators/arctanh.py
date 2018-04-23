"""Arc-Tangent Hyperbolic."""
import numpy

from ..baseclass import Dist
from .. import evaluation, approximation



class Arctanh(Dist):
    """
    Arc-Tangent Hyperbolic.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Arctanh(chaospy.Uniform(-0.5, 0.5))
        >>> print(distribution)
        Arctanh(Uniform(lower=-0.5, upper=0.5))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [-0.3095 -0.1003  0.1003  0.3095]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [10.1111 99.     99.     10.1111]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 0.1548 -0.4059  0.4851 -0.0178]
        >>> print(numpy.around(distribution.mom(2), 4))
        0.1006
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        return evaluation.evaluate_density(
            dist, numpy.tanh(x), cache=cache)/numpy.sinh(x)**2

    def _cdf(self, x, dist, cache):
        return evaluation.evaluate_forward(dist, numpy.tanh(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.arctanh(evaluation.evaluate_inverse(dist, q, cache=cache))

    def _bnd(self, x, dist, cache):
        return numpy.arctanh(evaluation.evaluate_bound(
            dist, numpy.tanh(x), cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"
