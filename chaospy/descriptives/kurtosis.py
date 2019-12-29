"""Kurtosis operator."""
import numpy

from .. import distributions, poly as polynomials
from .expected import E
from .standard_deviation import Std


def Kurt(poly, dist=None, fisher=True, **kws):
    """
    Kurtosis operator.

    Element by element 4rd order statistics of a distribution or polynomial.

    Args:
        poly (chaospy.poly.ndpoly, Dist):
            Input to take kurtosis on.
        dist (Dist):
            Defines the space the skewness is taken on. It is ignored if
            ``poly`` is a distribution.
        fisher (bool):
            If True, Fisher's definition is used (Normal -> 0.0). If False,
            Pearson's definition is used (normal -> 3.0)

    Returns:
        (numpy.ndarray):
            Element for element variance along ``poly``, where
            ``skewness.shape==poly.shape``.

    Examples:
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> chaospy.Kurt(dist).round(4)
        array([6., 0.])
        >>> chaospy.Kurt(dist, fisher=False).round(4)
        array([9., 3.])
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, x, y, 10*x*y])
        >>> chaospy.Kurt(poly, dist).round(4)
        array([nan,  6.,  0., 15.])
    """
    adjust = 3 if fisher else 0

    if dist is None:
        dist, poly = poly, polynomials.variable(len(poly))
    poly = polynomials.setdim(poly, len(dist))
    if not poly.isconstant:
        return poly.tonumpy()**4-adjust

    poly = poly-E(poly, dist, **kws)
    poly = poly/Std(poly, dist, **kws)
    return E(poly**4, dist, **kws)-adjust
