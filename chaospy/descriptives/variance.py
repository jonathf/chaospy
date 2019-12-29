"""Variance operator."""
import numpy

from .. import poly as polynomials
from .expected import E


def Var(poly, dist=None, **kws):
    """
    Element by element 2nd order statistics.

    Args:
        poly (chaospy.poly.ndpoly, Dist):
            Input to take variance on.
        dist (Dist):
            Defines the space the variance is taken on. It is ignored if
            ``poly`` is a distribution.

    Returns:
        (numpy.ndarray):
            Element for element variance along ``poly``, where
            ``variation.shape == poly.shape``.

    Examples:
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> chaospy.Var(dist)
        array([1., 4.])
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, x, y, 10*x*y])
        >>> chaospy.Var(poly, dist)
        array([  0.,   1.,   4., 800.])
    """
    if dist is None:
        dist, poly = poly, polynomials.variable(len(poly))
    poly = polynomials.setdim(poly, len(dist))
    if not poly.isconstant:

        return poly.tonumpy()**2
    poly = poly-E(poly, dist, **kws)
    poly = polynomials.square(poly)
    return E(poly, dist, **kws)
