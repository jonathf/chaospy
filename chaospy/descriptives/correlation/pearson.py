"""Pearson's correlation matrix."""
import numpy
import numpoly

from ... import distributions
from ..covariance import Cov


def Corr(poly, dist=None, **kws):
    """
    Correlation matrix of a distribution or polynomial.

    Args:
        poly (Poly, Dist):
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
        >>> print(chaospy.Corr(Z).round(4))
        [[1.     0.3536]
         [0.3536 1.    ]]

        >>> x = numpoly.symbols("x")
        >>> Z = chaospy.Normal()
        >>> print(chaospy.Corr([1, x, x**2], Z).round(4))
        [[0. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    """
    if isinstance(poly, distributions.Dist):
        dist, poly = poly, numpoly.symbols("q:%d" % len(poly))
    poly = numpoly.polynomial(poly)

    if not poly.shape:
        return numpy.ones((1, 1))

    cov = Cov(poly, dist, **kws)
    var = numpy.diag(cov)
    vvar = numpy.sqrt(numpy.outer(var, var))
    return numpy.where(vvar > 0, cov/vvar, 0)
