"""Laplace Probability Distribution."""
import numpy
from scipy import misc

from ..baseclass import Dist
from ..operators.addition import Add


class laplace(Dist):
    """Laplace Probability Distribution."""

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return numpy.e**-numpy.abs(x)/2

    def _cdf(self, x):
        return (1+numpy.sign(x)*(1-numpy.e**-abs(x)))/2

    def _mom(self, k):
        return .5*misc.factorial(k)*(1+(-1)**k)

    def _ppf(self, x):
        return numpy.where(x>.5, -numpy.log(2*(1-x)), numpy.log(2*x))

    def _bnd(self):
        return -32., 32.


class Laplace(Add):
    R"""
    Laplace Probability Distribution

    Args:
        mu (float, Dist) : Mean of the distribution.
        scale (float, Dist) : Scaleing parameter. scale > 0.

    Examples:
        >>> f = chaospy.Laplace(2, 2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(f.inv(q), 4))
        [0.1674 1.5537 2.4463 3.8326]
        >>> print(numpy.around(f.fwd(f.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(f.pdf(f.inv(q)), 4))
        [0.1 0.2 0.2 0.1]
        >>> print(numpy.around(f.sample(4), 4))
        [ 2.734  -0.9392  6.6165  1.9275]
        >>> print(f.mom(1))
        2.0
    """
    def __init__(self, mu=0, scale=1):
        self._repr = {"mu": mu, "scale": scale}
        Add.__init__(self, left=laplace()*scale, right=mu)
