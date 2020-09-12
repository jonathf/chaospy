"""Logarithm of another distribution."""
import numpy
import chaospy

from ..baseclass import Distribution
from .operator import OperatorDistribution


class Logn(OperatorDistribution):
    """
    Logarithm with base N.

    Args:
        dist (Distribution):
            Distribution to perform transformation on.
        base (int, float):
            the logarithm base.

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
        0.3516

    """

    def __init__(self, dist, base=2):
        assert isinstance(dist, Distribution)
        assert numpy.all(dist.lower > 0)
        assert base > 0 and base != 1
        super(Logn, self).__init__(
            left=dist,
            right=base,
            repr_args=[dist, base],
        )

    def _lower(self, left, right, cache):
        return numpy.log(left._get_lower(cache))/numpy.log(right.item(0))

    def _upper(self, left, right, cache):
        return numpy.log(left._get_upper(cache))/numpy.log(right.item(0))

    def _pdf(self, xloc, left, right, cache):
        base = right.item(0)
        return left._get_pdf(base**xloc, cache)*base**xloc*numpy.log(base)

    def _cdf(self, xloc, left, right, cache):
        return left._get_fwd(right.item(0)**xloc, cache)

    def _ppf(self, uloc, left, right, cache):
        return numpy.log(left._get_inv(uloc, cache))/numpy.log(right.item(0))

    def _mom(self, kloc, left, right, cache):
        raise chaospy.UnsupportedFeature("%s: Analytical moments for logarithm not supported", self)

    def _ttr(self, kloc, left, right, cache):
        raise chaospy.UnsupportedFeature("%s: Analytical TTR for logarithm not supported", self)

    def _value(self, left, right, cache):
        if isinstance(left, Distribution):
            return self
        return numpy.log(left)/numpy.log(right.item(0))



class Log(Logn):
    """
    Logarithm with base Euler's constant.

    Args:
        dist (Distribution):
            Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Log(chaospy.Uniform(1, 2))
        >>> distribution
        Log(Uniform(lower=1, upper=2))

    """

    def __init__(self, dist):
        super(Log, self).__init__(dist=dist, base=numpy.e)
        self._repr_args = [dist]


class Log10(Logn):
    """
    Logarithm with base 10.

    Args:
        dist (Distribution): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Log10(chaospy.Uniform(1, 2))
        >>> print(distribution)
        Log10(Uniform(lower=1, upper=2))

    """

    def __init__(self, dist):
        super(Log10, self).__init__(dist=dist, base=10)
        self._repr_args = [dist]
