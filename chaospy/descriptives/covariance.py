"""Covariance matrix."""
import numpy
import numpoly

from .expected import E
from .. import distributions


def Cov(poly, dist=None, **kws):
    """
    Covariance matrix, or 2rd order statistics.

    Args:
        poly (Poly, Dist) :
            Input to take covariance on. Must have ``len(poly) >= 2``.
        dist (Dist) :
            Defines the space the covariance is taken on.  It is ignored if
            `poly` is a distribution.

    Returns:
        (numpy.ndarray):
            Covariance matrix with shape ``poly.shape+poly.shape``.

    Examples:
        >>> dist = chaospy.MvNormal([0, 0], [[2, .5], [.5, 1]])
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([1, x, y, 10*x*y])
        >>> print(chaospy.Cov(poly, dist))
        [[  0.    0.    0.    0. ]
         [  0.    2.    0.5   0. ]
         [  0.    0.5   1.    0. ]
         [  0.    0.    0.  225. ]]
        >>> print(chaospy.Cov(dist))
        [[2.  0.5]
         [0.5 1. ]]
    """
    if isinstance(poly, distributions.Dist):
        dist, poly = poly, numpoly.symbols("q:%d" % len(poly))
    poly = numpoly.polynomial(poly)

    mu1 = E(poly, dist)
    mu2 = E(numpoly.outer(poly, poly), dist)
    return mu2-numpy.outer(mu1, mu1)
