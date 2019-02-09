"""Covariance matrix."""
import numpy

from .. import distributions, poly as polynomials


def Cov(poly, dist=None, **kws):
    """
    Covariance matrix, or 2rd order statistics.

    Args:
        poly (Poly, Dist) :
            Input to take covariance on. Must have `len(poly)>=2`.
        dist (Dist) :
            Defines the space the covariance is taken on.  It is ignored if
            `poly` is a distribution.

    Returns:
        (numpy.ndarray):
            Covariance matrix with shape ``poly.shape+poly.shape``.

    Examples:
        >>> dist = chaospy.MvNormal([0, 0], [[2, .5], [.5, 1]])
        >>> print(chaospy.Cov(dist))
        [[2.  0.5]
         [0.5 1. ]]
        >>> x, y = chaospy.variable(2)
        >>> poly = chaospy.Poly([1, x, y, 10*x*y])
        >>> print(chaospy.Cov(poly, dist))
        [[  0.    0.    0.    0. ]
         [  0.    2.    0.5   0. ]
         [  0.    0.5   1.    0. ]
         [  0.    0.    0.  225. ]]
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
