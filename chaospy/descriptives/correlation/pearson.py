"""Pearson's correlation matrix."""
import numpy
from scipy.stats import spearmanr
import numpoly
import chaospy


def Corr(poly, dist=None, **kws):
    """
    Correlation matrix of a distribution or polynomial.

    Args:
        poly (numpoly.ndpoly, Distribution):
            Input to take correlation on. Must have ``len(poly)>=2``.
        dist (Distribution):
            Defines the space the correlation is taken on.  It is ignored if
            ``poly`` is a distribution.

    Returns:
        (numpy.ndarray):
            Correlation matrix with
            ``correlation.shape == poly.shape+poly.shape``.

    Examples:
        >>> distribution = chaospy.MvNormal(
        ...     [3, 4], [[2, .5], [.5, 1]])
        >>> chaospy.Corr(distribution).round(4)
        array([[1.    , 0.3536],
               [0.3536, 1.    ]])
        >>> q0 = chaospy.variable()
        >>> poly = chaospy.polynomial([q0, q0**2])
        >>> distribution = chaospy.Normal()
        >>> chaospy.Corr(poly, distribution).round(4)
        array([[1., 0.],
               [0., 1.]])

    """
    if isinstance(poly, chaospy.Distribution):
        poly, dist = numpoly.variable(len(poly)), poly
    else:
        poly = numpoly.polynomial(poly)

    if not poly.shape:
        return numpy.ones((1, 1))

    cov = chaospy.Cov(poly, dist, **kws)
    var = numpy.diag(cov)
    vvar = numpy.sqrt(numpy.outer(var, var))
    return numpy.where(vvar > 0, cov/vvar, 0)
