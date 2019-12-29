"""Skewness operator."""
from .. import poly as polynomials

from .expected import E
from .standard_deviation import Std


def Skew(poly, dist=None, **kws):
    """
    Skewness operator.

    Element by element 3rd order statistics of a distribution or polynomial.

    Args:
        poly (chaospy.poly.ndpoly, Dist):
            Input to take skewness on.
        dist (Dist):
            Defines the space the skewness is taken on. It is ignored if
            ``poly`` is a distribution.

    Returns:
        (numpy.ndarray):
            Element for element variance along ``poly``, where
            ``skewness.shape == poly.shape``.

    Examples:
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> chaospy.Skew(dist)
        array([2., 0.])
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, x, y, 10*x*y])
        >>> chaospy.Skew(poly, dist)
        array([nan,  2.,  0.,  0.])
    """
    if dist is None:
        dist, poly = poly, polynomials.variable(len(poly))
    poly = polynomials.setdim(poly, len(dist))
    if not poly.isconstant:
        return poly.tonumpy()**3

    poly = poly-E(poly, dist, **kws)
    poly = poly/Std(poly, dist, **kws)
    return E(poly**3, dist, **kws)
