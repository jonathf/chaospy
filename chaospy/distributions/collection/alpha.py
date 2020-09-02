"""Alpha probability distribution."""
import numpy
from scipy import special

import chaospy
from ..baseclass import Dist
from ..operators import LocScaling


class alpha(Dist):
    """Standard Alpha distribution."""

    def __init__(self, a=1):
        a = numpy.atleast_1d(a)
        assert a.ndim == 1, "a has too many dimensions"
        Dist.__init__(self, a=a)

    def _cdf(self, x, a):
        return (special.ndtr(a-1./x.T)/special.ndtr(a)).T

    def _ppf(self, q, a):
        return 1.0/(a-special.ndtri(q.T*special.ndtr(a))).T

    def _pdf(self, x, a):
        return (1.0/(x.T**2)/special.ndtr(a)*
            numpy.e**(.5*(a-1.0/x.T)**2)/numpy.sqrt(2*numpy.pi)).T

    def _lower(self, a):
        return numpy.zeros(a.size)


class Alpha(LocScaling):
    """
    Alpha distribution.

    Args:
        shape (float, Dist):
            Shape parameter
        scale (float, Dist):
            Scale Parameter
        shift (float, Dist):
            Location of lower threshold

    Examples:
        >>> distribution = chaospy.Alpha(shape=[1, 2], scale=0.5)
        >>> distribution
        Alpha(scale=0.5, shape=[1, 2], shift=0)
        >>> mesh = numpy.mgrid[0.25:0.75:3j, 0.25:0.75:3j].reshape(2, -1)
        >>> mapped_mesh = distribution.inv(mesh)
        >>> mapped_mesh.round(2)
        array([[0.28, 0.28, 0.28, 0.42, 0.42, 0.42, 0.75, 0.75, 0.75],
               [0.19, 0.25, 0.36, 0.19, 0.25, 0.36, 0.19, 0.25, 0.36]])
        >>> numpy.allclose(distribution.fwd(mapped_mesh), mesh)
        True
        >>> distribution.pdf(mapped_mesh).round(2)
        array([32.15, 14.37,  8.04, 10.48,  4.68,  2.62,  3.34,  1.49,  0.84])
        >>> distribution.sample(4).round(4)
        array([[0.5717, 0.2174, 3.1229, 0.4037],
               [0.5251, 0.1776, 0.1332, 0.2189]])
    """

    def __init__(self, shape=1, scale=1, shift=0):
        self._repr = {"shape": shape, "scale": scale, "shift": shift}
        LocScaling.__init__(self, dist=alpha(shape), covariance=scale**2, mean=shift)
