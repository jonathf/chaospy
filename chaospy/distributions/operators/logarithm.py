"""Logarithm of another distribution."""
import numpy
import chaospy

from ..baseclass import Distribution, OperatorDistribution


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
        >>> distribution
        Logn(Uniform(lower=1, upper=2), 3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([0.166 , 0.3063, 0.4278, 0.535 ])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([1.3183, 1.5381, 1.7578, 1.9775])
        >>> distribution.sample(4).round(4)
        array([0.4578, 0.0991, 0.608 , 0.3582])
        >>> distribution.mom(1).round(4)
        0.3516

    """

    _operator = lambda self, left, right: (numpy.log(left).T/numpy.log(right).T).T

    def __init__(self, dist, base=2):
        assert isinstance(dist, Distribution)
        assert numpy.all(dist.lower > 0)
        assert base > 0 and base != 1
        super(Logn, self).__init__(
            left=dist,
            right=base,
            repr_args=[dist, base],
        )

    def _lower(self, idx, left, right, cache):
        return self._operator(left._get_lower(idx, cache), right)

    def _upper(self, idx, left, right, cache):
        return self._operator(left._get_upper(idx, cache), right)

    def _pdf(self, xloc, idx, left, right, cache):
        return left._get_pdf(right**xloc, idx, cache)*right**xloc*numpy.log(right)

    def _cdf(self, xloc, idx, left, right, cache):
        return left._get_fwd(right.item(0)**xloc, idx, cache)

    def _ppf(self, uloc, idx, left, right, cache):
        return self._operator(left._get_inv(uloc, idx, cache), right)

    def _mom(self, kloc, left, right, cache):
        raise chaospy.UnsupportedFeature("%s: Analytical moments for logarithm not supported", self)

    def _ttr(self, kloc, idx, left, right, cache):
        raise chaospy.UnsupportedFeature("%s: Analytical TTR for logarithm not supported", self)


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
        >>> distribution
        Log10(Uniform(lower=1, upper=2))

    """

    def __init__(self, dist):
        super(Log10, self).__init__(dist=dist, base=10)
        self._repr_args = [dist]
