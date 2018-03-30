"""Hyperbolic secant distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class hyperbolic_secant(Dist):
    """Hyperbolic secant distribution."""

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return .5*numpy.cosh(numpy.pi*x/2.)**-1

    def _cdf(self, x):
        return 2/numpy.pi*numpy.arctan(numpy.e**(numpy.pi*x/2.))

    def _ppf(self, q):
        return 2/numpy.pi*numpy.log(numpy.tan(numpy.pi*q/2.))

    def _mom(self, k):
        shape = k.shape
        output = numpy.abs([special.euler(k_)[-1] for k_ in k.flatten()])
        return output.reshape(shape)

    def _bnd(self):
        return self._ppf(1e-10), self._ppf(1-1e-10)


class HyperbolicSecant(Add):
    """
    Hyperbolic secant distribution

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scale parameter

    Examples:
        >>> distribution = chaospy.HyperbolicSecant(2, 2)
        >>> print(distribution)
        HyperbolicSecant(loc=2, scale=2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.5687 1.5933 2.4067 3.4313]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.1469 0.2378 0.2378 0.1469]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 2.6397 -0.1648  5.2439  1.9287]
        >>> print(distribution.mom(1))
        2.0
    """
    def __init__(self, loc=0, scale=1):
        self._repr = {"loc": loc, "scale": scale}
        Add.__init__(self, left=hyperbolic_secant()*scale, right=loc)
