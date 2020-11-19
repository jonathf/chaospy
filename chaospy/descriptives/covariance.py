"""Covariance matrix."""
import numpy
import numpoly

from .expected import E
from .. import distributions


def Cov(poly, dist=None, **kws):
    """
    Variance/Covariance matrix of a distribution or polynomial array.

    Args:
        poly (numpoly.ndpoly, Distribution) :
            Input to take covariance on. Must have `len(poly)>=2`.
        dist (Distribution) :
            Defines the space the covariance is taken on.  It is ignored if
            `poly` is a distribution.

    Returns:
        (numpy.ndarray):
            Covariance matrix with shape ``poly.shape+poly.shape``.

    Examples:
        >>> dist = chaospy.MvNormal([0, 0], [[2, .5], [.5, 1]])
        >>> chaospy.Cov(dist)
        array([[2. , 0.5],
               [0.5, 1. ]])
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0, q1, 10*q0*q1-1])
        >>> chaospy.Cov(poly, dist)
        array([[  0. ,   0. ,   0. ,   0. ],
               [  0. ,   2. ,   0.5,   0. ],
               [  0. ,   0.5,   1. ,   0. ],
               [  0. ,   0. ,   0. , 225. ]])
        >>> chaospy.Cov([1, 2, 3], dist)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])

    """
    if dist is None:
        dist, poly = poly, numpoly.variable(len(poly))
    poly = numpoly.set_dimensions(poly, len(dist))
    if poly.isconstant():
        return numpy.zeros((len(poly), len(poly)))
    poly = poly-E(poly, dist)
    poly = numpoly.outer(poly, poly)
    return E(poly, dist)
