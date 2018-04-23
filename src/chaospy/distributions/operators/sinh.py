"""Sinus Hyperbolic."""
import numpy
from ..baseclass import Dist
from .. import evaluation, approximation


class Sinh(Dist):
    """
    Sinus Hyperbolic.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Sinh(chaospy.Uniform(0, 1))
        >>> print(distribution)
        Sinh(Uniform(lower=0, upper=1))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.2013 0.4108 0.6367 0.8881]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.9803 0.925  0.8436 0.7477]
        >>> print(numpy.around(distribution.sample(4), 4))
        [0.7011 0.1153 1.0999 0.5011]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.5876
        >>> print(numpy.around(distribution.ttr([0, 1, 2]), 4))
        [[0.5431 0.5971 0.5885]
         [1.     0.1118 0.0927]]
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, numpy.arcsinh(x), cache=cache)/numpy.sqrt(1+x*x)

    def _cdf(self, x, dist, cache):
        return evaluation.evaluate_forward(dist, numpy.arcsinh(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.sinh(evaluation.evaluate_inverse(dist, q, cache=cache))

    def _bnd(self, x, dist, cache):
        return numpy.sinh(evaluation.evaluate_bound(
            dist, numpy.arcsinh(x), cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"
