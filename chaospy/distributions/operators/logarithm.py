"""Logarithm of another distribution."""
import numpy

from ..baseclass import Dist
from .unary import UnaryOperator


class Logn(UnaryOperator):
    """
    Logarithm with base N.

    Args:
        dist (Dist):
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
        assert isinstance(dist, Dist)
        assert numpy.all(dist.lower > 0)
        assert base > 0 and base != 1
        Dist.__init__(self, dist=dist, base=base)
        self._repr = {"_": [dist, base]}

    def _post_pdf(self, xloc, base):
        return base**xloc*numpy.log(base)

    def _pre_fwd(self, xloc, base):
        return base**xloc

    def _post_fwd(self, uloc, base):
        return uloc

    def _pre_inv(self, qloc, base):
        return qloc

    def _post_inv(self, uloc, base):
        return numpy.log(uloc)/numpy.log(base)


class Log(Logn):
    """
    Logarithm with base Euler's constant.

    Args:
        dist (Dist):
            Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Log(chaospy.Uniform(1, 2))
        >>> print(distribution)
        Log(Uniform(lower=1, upper=2))
    """

    def __init__(self, dist):
        super(Log, self).__init__(dist=dist, base=numpy.e)
        self._repr = {"_": [dist]}


class Log10(Logn):
    """
    Logarithm with base 10.

    Args:
        dist (Dist): Distribution to perform transformation on.

    Example:
        >>> distribution = chaospy.Log10(chaospy.Uniform(1, 2))
        >>> print(distribution)
        Log10(Uniform(lower=1, upper=2))
    """

    def __init__(self, dist):
        super(Log10, self).__init__(dist=dist, base=10)
        self._repr = {"_": [dist]}
