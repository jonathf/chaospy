"""Anglit distribution."""
import numpy

from ..baseclass import Dist
from ..operators import ShiftScale


class anglit(Dist):
    """Anglit distribution."""

    def __init__(self):
        super(anglit, self).__init__()

    def _pdf(self, x):
        return numpy.cos(2*x)

    def _cdf(self, x):
        return numpy.sin(x+numpy.pi/4)**2.0

    def _ppf(self, q):
        return (numpy.arcsin(numpy.sqrt(q))-numpy.pi/4)

    def _lower(self):
        return -numpy.pi/4

    def _upper(self):
        return numpy.pi/4


class Anglit(ShiftScale):
    """
    Anglit distribution.

    Args:
        loc (float, Dist):
            Location parameter
        scale (float, Dist):
            Scaling parameter

    Examples:
        >>> distribution = chaospy.Anglit(2, 4)
        >>> distribution
        Anglit(loc=2, scale=4)
        >>> q = numpy.linspace(0, 1, 5)
        >>> distribution.inv(q).round(4)
        array([-1.1416,  0.9528,  2.    ,  3.0472,  5.1416])
        >>> distribution.fwd(distribution.inv(q)).round(4)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> distribution.pdf(distribution.inv(q)).round(4)
        array([0.    , 0.2165, 0.25  , 0.2165, 0.    ])
        >>> distribution.sample(4).round(4)
        array([2.6245, 0.2424, 4.2421, 1.9288])
        >>> distribution.mom([1, 2, 3]).round(4)
        array([ 2.    ,  5.8696, 19.2176])
    """

    def __init__(self, loc=0, scale=1):
        self._repr = {"scale": scale, "loc": loc}
        super(Anglit, self).__init__(dist=anglit(), scale=scale, shift=loc)
