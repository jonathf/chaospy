"""Variance operator."""
import numpy
import numpoly

from .. import distributions
from .expected import E


def Var(poly, dist=None, **kws):
    """
    Element by element 2nd order statistics.

    Args:
        poly (Poly, Dist):
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
        >>> print(chaospy.Var(dist))
        [1. 4.]
        >>> x, y = numpoly.symbols("q:2")
        >>> poly = numpoly.polynomial([1, x, y, 10*x*y])
        >>> print(chaospy.Var(poly, dist))
        [  0.   1.   4. 800.]
    """
    if isinstance(poly, distributions.Dist):
        poly, dist = numpoly.symbols("q:%d" % len(poly)), poly
    poly = numpoly.polynomial(poly)
    return E(poly**2, dist)-E(poly, dist)**2
