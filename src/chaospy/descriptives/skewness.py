"""Skewness operator."""
import numpy

from .. import distributions, poly as polynomials
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
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.Poly([1, x, y, 10*x*y])
        >>> print(chaospy.Skew(poly, dist))
        [nan  2.  0.  0.]
    """
    if isinstance(poly, distributions.Dist):
        x = polynomials.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = polynomials.Poly(poly)

    if poly.dim < len(dist):
        polynomials.setdim(poly, len(dist))

    shape = poly.shape
    poly = polynomials.flatten(poly)

    m1 = E(poly, dist)
    m2 = E(poly**2, dist)
    m3 = E(poly**3, dist)
    out = (m3-3*m2*m1+2*m1**3)/(m2-m1**2)**1.5

    out = numpy.reshape(out, shape)
    return out
