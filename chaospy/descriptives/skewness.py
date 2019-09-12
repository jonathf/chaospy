"""Skewness operator."""
import numpy
import numpoly

from .. import distributions
from .expected import E


def Skew(poly, dist=None, **kws):
    """
    Skewness operator.

    Element by element 3rd order statistics of a distribution or polynomial.

    Args:
        poly (Poly, Dist):
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
        >>> print(chaospy.Skew(dist))
        [2. 0.]
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([1, x, y, 10*x*y])
        >>> print(chaospy.Skew(poly, dist))
        [nan  2.  0.  0.]
    """
    if isinstance(poly, distributions.Dist):
        poly, dist = numpoly.symbols("q:%d" % len(poly)), poly
    poly = numpoly.polynomial(poly)

    poly = poly[numpy.newaxis]
    poly = numpoly.concatenate([poly, poly**2, poly**3], axis=0)
    mu1, mu2, mu3 = E(poly, dist)
    return (mu3-3*mu2*mu1+2*mu1**3)*(mu2-mu1**2)**-1.5
