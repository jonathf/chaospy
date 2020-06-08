"""Fatigue-life distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class fatigue_life(Dist):
    """Fatigue-life distribution."""

    def __init__(self, c=0):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        output = (x+1)/(2*c*numpy.sqrt(2*numpy.pi*x**3))
        output *= numpy.exp(-(x-1)**2/(2.0*x*c**2))
        output[(x == 0) & numpy.isnan(output)] = 0
        return output

    def _cdf(self, x, c):
        return special.ndtr(1.0/c*(numpy.sqrt(x)-1.0/numpy.sqrt(x)))

    def _ppf(self, q, c):
        tmp = c*special.ndtri(q)
        return 0.25*(tmp + numpy.sqrt(tmp**2 + 4))**2

    def _lower(self, c):
        return 0.


class FatigueLife(Add):
    """
    Fatigue-Life or Birmbaum-Sanders distribution

    Args:
        shape (float, Dist):
            Shape parameter
        scale (float, Dist):
            Scaling parameter
        shift (float, Dist):
            Location parameter

    Examples:
        >>> distribution = chaospy.FatigueLife(2, 2, 1)
        >>> distribution
        FatigueLife(scale=2, shape=2, shift=1)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([ 1.4332,  2.2113,  4.3021, 10.2334])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.4223, 0.1645, 0.0603, 0.0198])
        >>> distribution.sample(4).round(4)
        array([ 5.3231,  1.2621, 26.5603,  2.8292])
        >>> distribution.mom(1).round(4)
        7.0
    """
    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=fatigue_life(shape)*scale, right=shift)
