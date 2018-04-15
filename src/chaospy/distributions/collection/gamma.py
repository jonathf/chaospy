"""Gamma distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class gamma(Dist):

    def __init__(self, a=1):
        Dist.__init__(self, a=a)

    def _pdf(self, x, a):
        return x**(a-1)*numpy.e**(-x) / special.gamma(a)

    def _cdf(self, x, a):
        return special.gammainc(a, x)

    def _ppf(self, q, a):
        return special.gammaincinv(a, q)

    def _mom(self, k, a):
        return special.gamma(a+k)/special.gamma(a)

    def _ttr(self, n, a):
        return 2.*n+a, n*n+n*(a-1)

    def _bnd(self, x, a):
        return 0, 40+2*a


class Gamma(Add):
    """
    Gamma distribution.

    Also an Erlang distribution when shape=k and scale=1./lamb.

    Args:
        shape (float, Dist) : Shape parameter. a>0
        scale () : Scale parameter. scale!=0
        shift (float, Dist) : Location of the lower bound.

    Examples:
        >>> distribution = chaospy.Gamma(1, 1, 1)
        >>> print(distribution)
        Gamma(scale=1, shape=1, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [1.2231 1.5108 1.9163 2.6094]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.8 0.6 0.4 0.2]
        >>> print(numpy.around(distribution.sample(4), 4))
        [2.0601 1.1222 4.0014 1.6581]
        >>> print(distribution.mom(1))
        2.0
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[4. 6. 8.]
         [1. 4. 9.]]
    """

    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=gamma(shape)*scale, right=shift)


class Exponential(Add):
    R"""
    Exponential Probability Distribution

    Args:
        scale (float, Dist) : Scale parameter. scale!=0
        shift (float, Dist) : Location of the lower bound.

    Examples;:
        >>> distribution = chaospy.Exponential(2, 3)
        >>> print(distribution)
        Exponential(scale=2, shift=3)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [3.4463 4.0217 4.8326 6.2189]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.sample(4), 4))
        [5.1203 3.2444 9.0028 4.3163]
        >>> print(numpy.around(distribution.mom(1), 4))
        5.0
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[ 9. 13. 17.]
         [ 4. 16. 36.]]
    """

    def __init__(self, scale=1, shift=0):
        self._repr = {"scale": scale, "shift": shift}
        Add.__init__(self, left=gamma(1)*scale, right=shift)
