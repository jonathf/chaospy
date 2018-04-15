"""Anglit distribution."""
import numpy

from ..baseclass import Dist
from ..operators.addition import Add


class anglit(Dist):
    """Anglit distribution."""

    def __init__(self):
        Dist.__init__(self)

    def _pdf(self, x):
        return numpy.cos(2*x)

    def _cdf(self, x):
        return numpy.sin(x+numpy.pi/4)**2.0

    def _ppf(self, q):
        return (numpy.arcsin(numpy.sqrt(q))-numpy.pi/4)

    def _bnd(self, x):
        return -numpy.pi/4, numpy.pi/4


class Anglit(Add):
    """
    Anglit distribution.

    Args:
        loc (float, Dist) : Location parameter
        scale (float, Dist) : Scaling parameter

    Examples:
        >>> distribution = chaospy.Anglit(2, 4)
        >>> print(distribution)
        Anglit(loc=2, scale=4)
        >>> q = numpy.linspace(0, 1, 5)
        >>> print(numpy.around(distribution.inv(q), 4))
        [-1.1416  0.9528  2.      3.0472  5.1416]
        >>> print(numpy.around(distribution.fwd(distribution.inv(q)), 4))
        [0.   0.25 0.5  0.75 1.  ]
        >>> print(numpy.around(distribution.pdf(distribution.inv(q)), 4))
        [0.     0.2165 0.25   0.2165 0.    ]
        >>> print(numpy.around(distribution.sample(4), 4))
        [2.6245 0.2424 4.2421 1.9288]
        >>> print(numpy.around(distribution.mom(1), 4))
        2.0
        >>> print(numpy.around(distribution.mom([1, 2, 3]), 4))
        [ 2.      7.2899 27.7392]
    """

    def __init__(self, loc=0, scale=1):
        self._repr = {"scale": scale, "loc": loc}
        Add.__init__(self, left=anglit()*scale, right=loc)
