"""Logarithm with base N."""
import numpy

from .. import evaluation, approximation
from ..baseclass import Dist


class Logn(Dist):
    """
    Logarithm with base N.

    Args:
        dist (Dist): Distribution to perform transformation on.
        base (int, float): the logarithm base.

    Example:
        >>> distribution = chaospy.Logn(chaospy.Uniform(1, 2), 3)
        >>> print(distribution)
        Logn(Uniform(lower=1, upper=2), 3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.166  0.3063 0.4278 0.535 ]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.3183 1.5381 1.7578 1.9775]
        >>> print(numpy.around(distribution.sample(4), 4))
        [0.4578 0.0991 0.608  0.3582]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.3155
        >>> print(numpy.around(distribution.ttr([0, 1, 2]), 4))
        [[0.3516 0.3085 0.3144]
         [1.     0.0324 0.0266]]
    """

    def __init__(self, dist, base=2):
        assert isinstance(dist, Dist)
        assert numpy.all(dist.range() > 0)
        assert base > 0 and base != 1
        Dist.__init__(self, dist=dist, base=base)

    def _pdf(self, xloc, dist, base, cache):
        """Probability density function."""
        return evaluation.evaluate_density(
            dist, base**xloc, cache=cache)*base**xloc*numpy.log(base)

    def _cdf(self, xloc, dist, base, cache):
        """Cumulative distribution function."""
        return evaluation.evaluate_forward(dist, base**xloc, cache=cache)

    def _ppf(self, q, dist, base, cache):
        """Point percentile function."""
        return numpy.log(evaluation.evaluate_inverse(
            dist, q, cache=cache))/numpy.log(base)

    def _bnd(self, xloc, dist, base, cache):
        """Distribution bounds."""
        return numpy.log(evaluation.evaluate_bound(
            dist, base**xloc, cache=cache)) / numpy.log(base)

    def _mom(self, x, dist, base, cache):
        return approximation.approximate_moment(
            self, x, params={"base": base}, cache=cache)

    def __len__(self):
        return len(self.prm["dist"])

    def __str__(self):
        return (self.__class__.__name__ + "(" + str(self.prm["dist"]) + ", " +
                str(self.prm["base"]) + ")")
