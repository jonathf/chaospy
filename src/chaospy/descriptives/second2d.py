"""Second order statistics."""
import numpy
from scipy.stats import spearmanr

from .. import distributions, poly as polynomials


def Cov(poly, dist=None, **kws):
    """
    Covariance matrix, or 2rd order statistics.

    Args:
        poly (Poly, Dist) : Input to take covariance on. Must have
                `len(poly)>=2`.
        dist (Dist) : Defines the space the covariance is taken on.  It is
                ignored if `poly` is a distribution.
        **kws (optional) : Extra keywords passed to dist.mom.

    Returns:
        (ndarray) : Covariance matrix with
                `covariance.shape==poly.shape+poly.shape`.

    Examples:
        >>> Z = chaospy.MvNormal([0, 0], [[2, .5], [.5, 1]])
        >>> print(numpy.around(chaospy.Cov(Z), 4))
        [[2.  0.5]
         [0.5 1. ]]

        >>> x = chaospy.variable()
        >>> Z = chaospy.Normal()
        >>> print(numpy.around(chaospy.Cov([x, x**2], Z), 4))
        [[1. 0.]
         [0. 2.]]
    """
    if not isinstance(poly, (distributions.Dist, polynomials.Poly)):
        poly = polynomials.Poly(poly)

    if isinstance(poly, distributions.Dist):
        x = polynomials.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = polynomials.Poly(poly)

    dim = len(dist)
    shape = poly.shape
    poly = polynomials.flatten(poly)
    keys = poly.keys
    N = len(keys)
    A = poly.A
    keys1 = numpy.array(keys).T
    if dim==1:
        keys1 = keys1[0]
        keys2 = sum(numpy.meshgrid(keys, keys))
    else:
        keys2 = numpy.empty((dim, N, N))
        for i in range(N):
            for j in range(N):
                keys2[:, i, j] = keys1[:, i]+keys1[:, j]

    m1 = dist.mom(keys1, **kws)
    m2 = dist.mom(keys2, **kws)
    mom = m2-numpy.outer(m1, m1)

    out = numpy.zeros((len(poly), len(poly)))
    for i in range(len(keys)):
        a = A[keys[i]]
        out += numpy.outer(a, a)*mom[i, i]
        for j in range(i+1, len(keys)):
            b = A[keys[j]]
            ab = numpy.outer(a, b)
            out += (ab+ab.T)*mom[i, j]

    out = numpy.reshape(out, shape+shape)
    return out



def Corr(poly, dist=None, **kws):
    """
    Correlation matrix of a distribution or polynomial.

    Args:
        poly (Poly, Dist) : Input to take correlation on. Must have
                `len(poly)>=2`.
        dist (Dist) : Defines the space the correlation is taken on.  It is
                ignored if `poly` is a distribution.
        **kws (optional) : Extra keywords passed to dist.mom.

    Returns:
        (ndarray) : Correlation matrix with
                `correlation.shape==poly.shape+poly.shape`.

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
        poly = polynomials.Poly(poly)

    cov = Cov(poly, dist, **kws)
    var = numpy.diag(cov)
    vvar = numpy.sqrt(numpy.outer(var, var))
    return numpy.where(vvar > 0, cov/vvar, 0)
