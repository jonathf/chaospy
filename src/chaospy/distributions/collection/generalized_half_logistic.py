"""Generalized half-logistic distribution."""
import numpy
from scipy import special

from ..baseclass import Dist
from ..operators.addition import Add


class generalized_half_logistic(Dist):
    """Generalized half-logistic distribution."""

    def __init__(self, c=1):
        Dist.__init__(self, c=c)

    def _pdf(self, x, c):
        limit = 1.0/c
        tmp = (1-c*x)
        tmp0 = tmp**(limit-1)
        tmp2 = tmp0*tmp
        return 2*tmp0 / (1+tmp2)**2

    def _cdf(self, x, c):
        limit = 1.0/c
        tmp = (1-c*x)
        tmp2 = tmp**(limit)
        return (1.0-tmp2) / (1+tmp2)

    def _ppf(self, q, c):
        return 1.0/c*(1-((1.0-q)/(1.0+q))**c)

    def _bnd(self, c):
        return 0.0, 1/numpy.where(c<10**-10, 10**-10, c)


class GeneralizedHalfLogistic(Add):
    """
    Generalized half-logistic distribution

    Args:
        shape (float, Dist) : Shape parameter
        scale (float, Dist) : Scaling parameter
        shift (float, Dist) : Location parameter

    Examples:
        >>> distribution = chaospy.GeneralizedHalfLogistic(3, 2, 2)
        >>> print(distribution)
        GeneralizedHalfLogistic(scale=2, shape=3, shift=2)
        >>> q = numpy.linspace(0, 1, 6)[1:-1]
        >>> print(numpy.around(distribution.inv(q), 4))
        [2.4691 2.6142 2.6562 2.6658]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.2 0.4 0.6 0.8]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [ 0.81    2.6678 10.24   65.61  ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [2.6605 2.3333 2.6667 2.6382]
        >>> print(numpy.around(distribution.mom(1), 4))
        2.3333
    """

    def __init__(self, shape, scale, shift):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        Add.__init__(
            self, left=generalized_half_logistic(shape)*scale, right=shift)
