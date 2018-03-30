"""Gompertz distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class gompertz(Dist):
    """Gompertz distribution."""

    def __init__(self, c):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        ex = numpy.exp(x)
        return c*ex*numpy.exp(-c*(ex-1))

    def _cdf(self, x, c):
        return 1.0-numpy.exp(-c*(numpy.exp(x)-1))

    def _ppf(self, q, c):
        return numpy.log(1-1.0/c*numpy.log(1-q))

    def _bnd(self, c):
        return 0., self._ppf(1-1e-10, c)


class Gompertz(Add):
    """
    Gompertz distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.Gompertz(3, 2, 2)
        >>> print(distribution)
        Gompertz(scale=2, shape=3, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [2.1435 2.3145 2.5331 2.859 ]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [1.2893 1.0532 0.7833 0.4609]
        >>> print(numpy.around(distribution.sample(4), 4))
        [2.6052 2.0798 3.3868 2.3967]
        >>> print(numpy.around(distribution.mom(1), 4))
        4.1605
    """

    def __init__(self, shape, scale, shift):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(self, left=gompertz(shape)*scale, right=shift)
