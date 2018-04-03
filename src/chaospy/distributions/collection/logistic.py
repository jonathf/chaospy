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
        return -numpy.log(q**(-1/c)-1)

    def _bnd(self, x, c):
        return self._ppf(1e-10, c), self._ppf(1-1e-10, c)


class Logistic(Add):
    """
    Generalized logistic type 1 distribution
    Sech squared distribution

    Args:
        loc (float, Dist): Location parameter
        scale (float, Dist): Scale parameter
        skew (float, Dist): Shape parameter

    Examples:
        >>> f = chaospy.Logistic(2, 2, 2)
        >>> print(f)
        Logistic(loc=2, scale=2, skew=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [1.5761 3.0855 4.4689 6.2736]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.pdf(f.inv(q)), 4))
        [0.0553 0.0735 0.0676 0.0422]
        >>> print(numpy.around(f.sample(4), 4))
        [4.8799 0.6656 9.3128 3.6415]
        >>> print(numpy.around(f.mom(1), 4))
        14.2061
    """
    def __init__(self, loc=0, scale=1, skew=1):
        self._repr = {"loc": loc, "scale": scale, "skew": skew}
        Add.__init__(self, left=logistic(skew)*scale, right=loc)
