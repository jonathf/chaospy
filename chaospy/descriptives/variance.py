"""Variance operator."""
import numpy
import numpoly

from .expected import E


def Var(poly, dist=None, **kws):
    """
    The variance of a distribution or polynomial.

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
        >>> chaospy.Var(dist)
        array([1., 4.])
        >>> q0, q1 = chaospy.variable(2)
        >>> poly = chaospy.polynomial([1, q0, q1, 10*q0*q1])
        >>> chaospy.Var(poly, dist)
        array([  0.,   1.,   4., 800.])
        >>> chaospy.Var(2., dist)
        array(0.)

    """
    if dist is None:
        dist, poly = poly, numpoly.variable(len(poly))
    poly = numpoly.set_dimensions(poly, len(dist))
    if poly.isconstant():
        return numpy.zeros(poly.shape)
    poly = poly-E(poly, dist, **kws)
    poly = numpoly.square(poly)
    return E(poly, dist, **kws)
