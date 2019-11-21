"""Standard deviation."""
import numpy

from .variance import Var


def Std(poly, dist=None, **kws):
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
        >>> print(chaospy.Std(dist))
        [1. 2.]
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, x, y, 10*x*y])
        >>> print(chaospy.Std(poly, dist))
        [ 0.          1.          2.         28.28427125]
    """
    return numpy.sqrt(Var(poly, dist=dist, **kws))
