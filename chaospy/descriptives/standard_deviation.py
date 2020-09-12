"""Standard deviation."""
import numpy

from .variance import Var


def Std(poly, dist=None, **kws):
    """
    Element by element 2nd order statistics.

    Args:
        poly (numpoly.ndpoly, Distribution):
            Input to take variance on.
        dist (Distribution):
            Defines the space the variance is taken on. It is ignored if
            ``poly`` is a distribution.

    Returns:
        (numpy.ndarray):
            Element for element variance along ``poly``, where
            ``variation.shape == poly.shape``.

    Examples:
        >>> dist = chaospy.J(chaospy.Gamma(1, 1), chaospy.Normal(0, 2))
        >>> chaospy.Std(dist)
        array([1., 2.])
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0, q1, 10*q0*q1-1])
        >>> chaospy.Std(poly, dist)
        array([ 0.        ,  1.        ,  2.        , 28.28427125])

    """
    return numpy.sqrt(Var(poly, dist=dist, **kws))
