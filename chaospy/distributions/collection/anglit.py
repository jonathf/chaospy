"""Anglit distribution."""
import numpy

from ..baseclass import SimpleDistribution, ShiftScaleDistribution


class anglit(SimpleDistribution):
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


class Anglit(ShiftScaleDistribution):
    """
    Anglit distribution.

    Args:
        loc (float, Distribution):
            Location parameter
        scale (float, Distribution):
            Scaling parameter

    Examples:
        >>> distribution = chaospy.Anglit()
        >>> distribution
        Anglit()
        >>> uloc = numpy.linspace(0, 1, 6)
        >>> uloc
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        >>> xloc = distribution.inv(uloc)
        >>> xloc.round(3)
        array([-0.785, -0.322, -0.101,  0.101,  0.322,  0.785])
        >>> numpy.allclose(distribution.fwd(xloc), uloc)
        True
        >>> distribution.pdf(xloc).round(3)
        array([0.  , 0.8 , 0.98, 0.98, 0.8 , 0.  ])
        >>> distribution.sample(4).round(3)
        array([ 0.156, -0.439,  0.561, -0.018])

    """

    def __init__(self, scale=1, shift=0):
        super(Anglit, self).__init__(dist=anglit(), scale=scale, shift=shift)
