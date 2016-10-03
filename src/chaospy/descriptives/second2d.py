import numpy as np
from scipy.stats import spearmanr

import chaospy as cp


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
        >>> Z = cp.MvNormal([0, 0], [[2, .5], [.5, 1]])
        >>> print(cp.Cov(Z))
        [[ 2.   0.5]
         [ 0.5  1. ]]

        >>> x = cp.variable()
        >>> Z = cp.Normal()
        >>> print(cp.Cov([x, x**2], Z))
        [[ 1.  0.]
         [ 0.  2.]]
    """
    if not isinstance(poly, (cp.dist.Dist, cp.poly.Poly)):
        poly = cp.poly.Poly(poly)

    if isinstance(poly, cp.dist.Dist):
        x = cp.poly.variable(len(poly))
        poly, dist = x, poly
    else:
        poly = cp.poly.Poly(poly)

    dim = len(dist)
    shape = poly.shape
    poly = cp.poly.flatten(poly)
    keys = poly.keys
    N = len(keys)
    A = poly.A
    keys1 = np.array(keys).T
    if dim==1:
        keys1 = keys1[0]
        keys2 = sum(np.meshgrid(keys, keys))
    else:
        keys2 = np.empty((dim, N, N))
        for i in range(N):
            for j in range(N):
                keys2[:, i, j] = keys1[:, i]+keys1[:, j]

    m1 = dist.mom(keys1, **kws)
    m2 = dist.mom(keys2, **kws)
    mom = m2-np.outer(m1, m1)

    out = np.zeros((len(poly), len(poly)))
    for i in range(len(keys)):
        a = A[keys[i]]
        out += np.outer(a, a)*mom[i, i]
        for j in range(i+1, len(keys)):
            b = A[keys[j]]
            ab = np.outer(a, b)
            out += (ab+ab.T)*mom[i, j]

    out = np.reshape(out, shape+shape)
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
        >>> Z = cp.MvNormal([3, 4], [[2, .5], [.5, 1]])
        >>> print(cp.Corr(Z))
        [[ 1.          0.35355339]
         [ 0.35355339  1.        ]]

        >>> x = cp.variable()
        >>> Z = cp.Normal()
        >>> print(cp.Corr([x, x**2], Z))
        [[ 1.  0.]
         [ 0.  1.]]
    """
    if isinstance(poly, cp.dist.Dist):
        poly, dist = cp.poly.variable(len(poly)), poly
    else:
        poly = cp.poly.Poly(poly)

    cov = Cov(poly, dist, **kws)
    var = np.diag(cov)
    vvar = np.sqrt(np.outer(var, var))
    return np.where(vvar > 0, cov/vvar, 0)
