"""Kurtosis operator."""
import numpy

from .. import distributions, poly as polynomials
from .expected import E


def Kurt(poly, dist=None, fisher=True, **kws):
    """
    Kurtosis operator.

    Element by element 4rd order statistics of a distribution or polynomial.

    Args:
        poly (Poly, Dist):
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
        >>> print(numpy.around(chaospy.Kurt(dist), 4))
        [6. 0.]
        >>> print(numpy.around(chaospy.Kurt(dist, fisher=False), 4))
        [9. 3.]
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.Poly([1, x, y, 10*x*y])
        >>> print(numpy.around(chaospy.Kurt(poly, dist), 4))
        [nan  6.  0. 15.]
    """
    if isinstance(poly, distributions.Dist):
        x = polynomials.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = polynomials.Poly(poly)

    if fisher:
        adjust = 3
    else:
        adjust = 0

    shape = poly.shape
    poly = polynomials.flatten(poly)

    m1 = E(poly, dist)
    m2 = E(poly**2, dist)
    m3 = E(poly**3, dist)
    m4 = E(poly**4, dist)

    out = (m4-4*m3*m1 + 6*m2*m1**2 - 3*m1**4) /\
            (m2**2-2*m2*m1**2+m1**4) - adjust

    out = numpy.reshape(out, shape)
    return out
