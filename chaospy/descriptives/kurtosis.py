"""Kurtosis operator."""
import numpy
import numpoly

from .. import distributions
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
        >>> x, y = numpoly.symbols("x y")
        >>> poly = numpoly.polynomial([1, x, y, 10*x*y])
        >>> print(numpy.around(chaospy.Kurt(poly, dist), 4))
        [nan  6.  0. 15.]
    """
    if isinstance(poly, distributions.Dist):
        poly, dist = numpoly.symbols("q:%d" % len(poly)), poly
    poly = numpoly.polynomial(poly)

    if fisher:
        adjust = 3
    else:
        adjust = 0

    poly = poly[numpy.newaxis]
    poly = numpoly.concatenate([poly, poly**2, poly**3, poly**4], axis=0)
    mu1, mu2, mu3, mu4 = E(poly, dist)
    return ((mu4-4*mu3*mu1+6*mu2*mu1**2-3*mu1**4)/
            (mu2**2-2*mu2*mu1**2+mu1**4)-adjust)
