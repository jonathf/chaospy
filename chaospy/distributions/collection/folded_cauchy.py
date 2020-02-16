"""Folded Cauchy distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add
from .deprecate import deprecation_warning


class folded_cauchy(Dist):
    """Folded Cauchy distribution."""

    def __init__(self, c=0):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        return 1./(numpy.pi*(1+(x-c)**2)) + 1./(numpy.pi*(1+(x+c)**2))

    def _cdf(self, x, c):
        return (numpy.arctan(x-c) + numpy.arctan(x+c))/numpy.pi

    def _lower(self, c):
        return 0.

    def _upper(self, c):
        return 10.**10  # actually infinity


class FoldedCauchy(Add):
    """
    Folded Cauchy distribution.

    Args:
        shape (float, Dist):
            Shape parameter
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.FoldedCauchy(3, 2, 1)
        >>> distribution
        FoldedCauchy(scale=2, shape=3, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([ 5.1449,  6.708 ,  8.0077, 10.6502])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0915, 0.1603, 0.1306, 0.0393])
        >>> distribution.sample(4).round(4)
        array([9.2648, 7.5918, 7.2159, 7.1782])
    """

    def __init__(self, shape=0, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=folded_cauchy(shape)*scale, right=shift)


Foldcauchy = deprecation_warning(FoldedCauchy, "Foldcauchy")
