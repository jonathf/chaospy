"""Pareto type 1 distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class pareto1(Dist):
    """Pareto type 1 distribution."""

    def __init__(self, b):
        Dist.__init__(self, b=b)

    def _pdf(self, x, b):
        return b * x**(-b-1)

    def _cdf(self, x, b):
        return 1 -  x**(-b)

    def _ppf(self, q, b):
        return pow(1-q, -1.0/b)

    def _bnd(self, x, b):
        return 1.0, self._ppf(1-1e-10, b)


class Pareto1(Add):
    """
    Pareto type 1 distribution.

    Lower threshold at scale+loc and survival: x^-shape

    Args:
        shape (float, Dist): Tail index parameter
        scale (float, Dist): Scaling parameter
        shift (float, Dist): Location parameter

    Examples:
        >>> f = chaospy.Pareto1(2, 2, 2)
        >>> print(f)
        Pareto1(loc=2, scale=2, shape=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [4.2361 4.582  5.1623 6.4721]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.pdf(f.inv(q)), 4))
        [0.7155 0.4648 0.253  0.0894]
        >>> print(numpy.around(f.sample(4), 4))
        [ 5.3981  4.126  10.9697  4.7794]
        >>> print(numpy.around(f.mom(1), 4))
        100002.9959
    """

    def __init__(self, shape=1, scale=1, loc=0):
        self._repr = {"shape": shape, "scale": scale, "loc": loc}
        Add.__init__(self, left=pareto1(shape)*scale, right=loc)
