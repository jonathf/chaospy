"""Skewness operator."""
import numpoly

from .expected import E
from .standard_deviation import Std


def Skew(poly, dist=None, **kws):
    """
    Skewness operator.

    Element by element 3rd order statistics of a distribution or polynomial.

    Args:
        poly (numpoly.ndpoly, Distribution):
            Input to take skewness on.
        dist (Distribution):
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
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0, q1, 10*q0*q1-1])
        >>> chaospy.Skew(poly, dist)
        array([nan,  2.,  0.,  0.])
    """
    if dist is None:
        dist, poly = poly, numpoly.variable(len(poly))
    poly = numpoly.set_dimensions(poly, len(dist))
    if not poly.isconstant:
        return poly.tonumpy()**3

    poly = poly-E(poly, dist, **kws)
    poly = numpoly.true_divide(poly, Std(poly, dist, **kws))
    return E(poly**3, dist, **kws)
