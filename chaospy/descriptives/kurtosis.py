"""Kurtosis operator."""
import numpy
import numpoly

from .. import distributions
from .expected import E
from .standard_deviation import Std


def Kurt(poly, dist=None, fisher=True, **kws):
    """
    Kurtosis operator.

    Element by element 4rd order statistics of a distribution or polynomial.

    Args:
        poly (numpoly.ndpoly, Distribution):
            Input to take kurtosis on.
        dist (Distribution):
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
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0, q1, 10*q0*q1-1])
        >>> chaospy.Kurt(poly, dist).round(4)
        array([nan,  6.,  0., 15.])
    """
    adjust = 3 if fisher else 0

    if dist is None:
        dist, poly = poly, numpoly.variable(len(poly))
    poly = numpoly.set_dimensions(poly, len(dist))
    if not poly.isconstant:
        return poly.tonumpy()**4-adjust

    poly = poly-E(poly, dist, **kws)
    poly = numpoly.true_divide(poly, Std(poly, dist, **kws))
    return E(poly**4, dist, **kws)-adjust
