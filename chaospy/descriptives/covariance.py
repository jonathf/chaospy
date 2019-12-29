"""Covariance matrix."""
import numpy

from .expected import E
from .. import distributions, poly as polynomials


def Cov(poly, dist=None, **kws):
    """
    Covariance matrix, or 2rd order statistics.

    Args:
        poly (chaospy.poly.ndpoly, Dist) :
            Input to take covariance on. Must have `len(poly)>=2`.
        dist (Dist) :
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
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, x, y, 10*x*y])
        >>> chaospy.Cov(poly, dist)
        array([[  0. ,   0. ,   0. ,   0. ],
               [  0. ,   2. ,   0.5,   0. ],
               [  0. ,   0.5,   1. ,   0. ],
               [  0. ,   0. ,   0. , 225. ]])
    """
    if dist is None:
        dist, poly = poly, polynomials.variable(len(poly))
    poly = polynomials.setdim(poly, len(dist))
    if not poly.isconstant:
        return poly.tonumpy()**2
    poly = poly-E(poly, dist)
    poly = polynomials.outer(poly, poly)
    return E(poly, dist)
