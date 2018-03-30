"""Raised cosine distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class raised_cosine(Dist):
    """Raised cosine distribution."""

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return .5 + .5*numpy.cos(numpy.pi*x)

    def _cdf(self, x):
        return .5 + .5*x + numpy.sin(numpy.pi*x)/(2*numpy.pi)

    def _bnd(self):
        return -1,1

    def _mom(self, k):
        output = numpy.array([special.hyp1f2(k_+.5, .5, k_+1.5, -numpy.pi**2/4)[0]
                              for k_ in k.flatten()]).reshape(k.shape)
        output = 1/(1.+k) + 1/(1.+2*k)*output
        output = numpy.where(k % 2, output, 0)
        return output


class RaisedCosine(Add):
    """
    Raised cosine distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter

    Examples:
        >>> distribution = chaospy.RaisedCosine(2, 2)
        >>> print(distribution)
        RaisedCosine(loc=2, scale=2)
        >>> q = numpy.linspace(0, 1, 7)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [1.2535 1.6586 2.     2.3414 2.7465]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.1667 0.3333 0.5    0.6667 0.8333]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.3469 0.4649 0.5    0.4649 0.3469]
        >>> print(numpy.around(distribution.sample(4), 4))
        [2.3134 1.0903 3.1938 1.9644]
        >>> print(numpy.around(distribution.mom(1), 4))
        0.5947
    """

    def __init__(self, loc=0, scale=1):
        self._repr = {"loc": loc, "scale": scale}
        Add.__init__(self, left=raised_cosine()*scale, right=loc)
