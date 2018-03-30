"""Folded Cauchy distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add


class folded_cauchy(Dist):
    """Folded Cauchy distribution."""

    def __init__(self, c=0):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        return 1./(numpy.pi*(1+(x-c)**2)) + 1./(numpy.pi*(1+(x+c)**2))

    def _cdf(self, x, c):
        return (numpy.arctan(x-c) + numpy.arctan(x+c))/numpy.pi

    def _bnd(self, c):
        return 0, 10**10


class FoldedCauchy(Add):
    """
    Folded Cauchy distribution.

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.FoldedCauchy(3, 2, 1)
        >>> print(distribution)
        FoldedCauchy(scale=2, shape=3, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [ 5.145   6.708   8.0077 10.6503]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.0915 0.1603 0.1306 0.0393]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 8.4584  3.9544 27.8886  7.2135]
        >>> print(distribution.mom(1) > 10**9) # undefined
        True
    """

    def __init__(self, shape=0, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=folded_cauchy(shape)*scale, right=shift)
