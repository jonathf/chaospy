"""Pearson's correlation matrix."""
import numpy
from scipy.stats import spearmanr

from ... import distributions, poly as polynomials
from ..covariance import Cov


def Corr(poly, dist=None, **kws):
    """
    Correlation matrix of a distribution or polynomial.

    Args:
        poly (chaospy.poly.ndpoly, Dist):
            Input to take correlation on. Must have ``len(poly)>=2``.
        dist (Dist):
            Defines the space the correlation is taken on.  It is ignored if
            ``poly`` is a distribution.

    Returns:
        (numpy.ndarray):
            Correlation matrix with
            ``correlation.shape == poly.shape+poly.shape``.

    Examples:
        >>> Z = chaospy.MvNormal([3, 4], [[2, .5], [.5, 1]])
        >>> print(numpy.around(chaospy.Corr(Z), 4))
        [[1.     0.3536]
         [0.3536 1.    ]]

        >>> x = chaospy.variable()
        >>> Z = chaospy.Normal()
        >>> print(numpy.around(chaospy.Corr([x, x**2], Z), 4))
        [[1. 0.]
         [0. 1.]]
    """
    if isinstance(poly, distributions.Dist):
        poly, dist = polynomials.variable(len(poly)), poly
    else:
        poly = polynomials.polynomial(poly)

    if not poly.shape:
        return numpy.ones((1, 1))

    cov = Cov(poly, dist, **kws)
    var = numpy.diag(cov)
    vvar = numpy.sqrt(numpy.outer(var, var))
    return numpy.where(vvar > 0, cov/vvar, 0)
