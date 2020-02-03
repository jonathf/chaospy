"""Generalized logistic type 1 distribution."""
import numpy
from scipy import special, misc

from ..baseclass import Dist
from ..operators.addition import Add

class logistic(Dist):
    """Generalized logistic type 1 distribution."""

    def __init__(self, c=1):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        return numpy.e**-x/(1+numpy.e**-x)**(c+1)

    def _cdf(self, x, c):
        return (1+numpy.e**-x)**-c

    def _ppf(self, q, c):
        return -numpy.log(q**(-1./c)-1)


class Logistic(Add):
    """
    Generalized logistic type 1 distribution
    Sech squared distribution

    Args:
        loc (float, Dist):
            Location parameter
        scale (float, Dist):
            Scale parameter
        skew (float, Dist):
            Shape parameter

    Examples:
        >>> distribution = chaospy.Logistic(2, 2, 2)
        >>> distribution
        Logistic(loc=2, scale=2, skew=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> distribution.inv(q).round(4)
        array([1.5761, 3.0855, 4.4689, 6.2736])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.2, 0.4, 0.6, 0.8])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.0553, 0.0735, 0.0676, 0.0422])
        >>> distribution.sample(4).round(4)
        array([4.8799, 0.6656, 9.3128, 3.6415])
        >>> distribution.mom(1).round(4)
        4.0
    """
    def __init__(self, loc=0, scale=1, skew=1):
        self._repr = {"loc": loc, "scale": scale, "skew": skew}
        Add.__init__(self, left=logistic(skew)*scale, right=loc)
