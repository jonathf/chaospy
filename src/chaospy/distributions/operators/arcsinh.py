"""Arc-Sinus Hyperbolic."""
import numpy
from ..baseclass import Dist
from .. import evaluation, approximation


class Arcsinh(Dist):
    """
    Arc-Sinus Hyperbolic.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Arcsinh(chaospy.Uniform(0, 1))
        >>> print(distribution)
        Arcsinh(Uniform(lower=0, upper=1))
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.1987 0.39   0.5688 0.7327]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.5907 1.7404 2.0115 2.4324]
        >>> print(numpy.around(distribution.sample(4), 4))
        [0.6142 0.1148 0.8458 0.4652]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.4407
        >>> print(numpy.around(distribution.ttr([0, 1, 2]), 4))
        [[0.8479 0.2254 0.5361]
         [1.     0.3332 0.7026]]
    """

    def __init__(self, dist):
        assert isinstance(dist, Dist)
        Dist.__init__(self, dist=dist)

    def _pdf(self, x, dist, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, numpy.sinh(x), cache=cache)*numpy.cosh(1+x*x)

    def _cdf(self, x, dist, cache):
        return evaluation.evaluate_forward(dist, numpy.sinh(x), cache=cache)

    def _ppf(self, q, dist, cache):
        return numpy.arcsinh(evaluation.evaluate_inverse(dist, q, cache=cache))

    def _bnd(self, x, dist, cache):
        return numpy.arcsinh(evaluation.evaluate_bound(
            dist, numpy.sinh(x), cache=cache))

    def _mom(self, x, dist, cache):
        return approximation.approximate_moment(self, x, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.prm["dist"]) + ")"
