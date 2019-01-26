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

    def _bnd(self, x):
        return -32., 32.


class Laplace(Add):
    R"""
    Laplace Probability Distribution

    Args:
        mu (float, Dist) : Mean of the distribution.
        scale (float, Dist) : Scaleing parameter. scale > 0.

    Examples:
        >>> distribution = chaospy.Laplace(2, 2)
        >>> q = numpy.linspace(0,1,6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [0.1674 1.5537 2.4463 3.8326]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.1 0.2 0.2 0.1]
        >>> print(numpy.around(distribution.sample(4), 4))
        [ 2.734  -0.9392  6.6165  1.9275]
        >>> print(numpy.around(distribution.mom(1), 4))
        2.0
        >>> print(numpy.around(distribution.ttr([1, 2, 3]), 4))
        [[ 2.      2.      2.    ]
         [ 7.9685 40.888  84.6276]]
    """
    def __init__(self, mu=0, scale=1):
        self._repr = {"mu": mu, "scale": scale}
        Add.__init__(self, left=laplace()*scale, right=mu)
