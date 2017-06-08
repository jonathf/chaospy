"""
Miscellenious descriptive statistics.
"""
import numpy as np
from scipy.stats import spearmanr

import chaospy as cp


def Acf(poly, dist, N=None, **kws):
    """
    Auto-correlation function.

    Args:
        poly (Poly) : Polynomial of interest. Must have `len(poly)>N`
        dist (Dist) : Defines the space the correlation is taken on.
        N (int, optional) : The number of time steps appart included. If omited
                set to `len(poly)/2+1`.
        **kws (optional) : Extra keywords passed to dist.mom.

    Returns:
        (ndarray) : Auto-correlation of `poly` with `shape=(N, )`.
                Note that by definition `Q[0]=1`.

    Examples:
        >>> poly = cp.prange(10)[1:]
        >>> Z = cp.Uniform()
        >>> print(cp.Acf(poly, Z, 5))
        [ 1.          0.99148391  0.9721971   0.94571181  0.91265479]
    """
    if N is None:
        N = len(poly)/2 + 1

    corr = cp.descriptives.Corr(poly, dist, **kws)
    out = np.empty(N)

    for n in range(N):
        out[n] = np.mean(corr.diagonal(n), 0)

    return out


def Spearman(poly, dist, sample=10000, retall=False, **kws):
    """
    Calculate Spearman's rank-order correlation coefficient.

    Args:
        poly (Poly) : Polynomial of interest.
        dist (Dist) : Defines the space where correlation is taken.
        sample (int) : Number of samples used in estimation.
        retall (bool) : If true, return p-value as well.
        **kws (optional) : Extra keywords passed to dist.sample.

    Returns:
        rho (float or ndarray) : Correlation output. Of type float if
                two-dimensional problem. Correleation matrix if larger.
        p-value (float or ndarray) : The two-sided p-value for a hypothesis
                test whose null hypothesis is that two sets of data are
                uncorrelated, has same dimension as rho.
    """
    samples = dist.sample(sample, **kws)
    poly = cp.poly.flatten(poly)
    Y = poly(*samples)
    if retall:
        return spearmanr(Y.T)
    return spearmanr(Y.T)[0]


def Perc(poly, q, dist, sample=10000, **kws):
    """
    Percentile function.

    Note that this function is an empirical function that operates using Monte
    Carlo sampling.

    Args:
        poly (Poly) : Polynomial of interest.
        q (array_like) : positions where percentiles are taken. Must be
                a number or an array, where all values are on the interval
                `[0, 100]`.
        dist (Dist) : Defines the space where percentile is taken.
        sample (int) : Number of samples used in estimation.
        **kws (optional) : Extra keywords passed to dist.sample.

    Returns:
        (ndarray) : Percentiles of `poly` with `Q.shape=poly.shape+q.shape`.

    Examples:
        >>> cp.seed(1000)
        >>> x, y = cp.variable(2)
        >>> poly = cp.Poly([x, x*y])
        >>> Z = cp.J(cp.Uniform(3, 6), cp.Normal())
        >>> print(cp.Perc(poly, [0, 50, 100], Z))
        [[  3.         -45.        ]
         [  4.5080777   -0.05862173]
         [  6.          45.        ]]
    """
    shape = poly.shape
    poly = cp.poly.flatten(poly)

    q = np.array(q)/100.
    dim = len(dist)

    # Interior
    Z = dist.sample(sample, **kws)
    if dim==1:
        Z = (Z, )
        q = np.array([q])
    poly1 = poly(*Z)

    # Min/max
    mi, ma = dist.range().reshape(2, dim)
    ext = np.mgrid[(slice(0, 2, 1), )*dim].reshape(dim, 2**dim).T
    ext = np.where(ext, mi, ma).T
    poly2 = poly(*ext)
    poly2 = np.array([_ for _ in poly2.T if not np.any(np.isnan(_))]).T

    # Finish
    if poly2.shape:
        poly1 = np.concatenate([poly1, poly2], -1)
    samples = poly1.shape[-1]
    poly1.sort()
    out = poly1.T[np.asarray(q*(samples-1), dtype=int)]
    out = out.reshape(q.shape + shape)
    return out


def QoI_Dist(poly, dist, sample=10000, **kws):
    """
    Constructs distributions for the quantity of interests.

    The function constructs a kernel density estimator (KDE) for each
    polynomial (poly) by sampling it.  With the KDEs, distributions (Dists) are
    constructed.  The Dists can be used for e.g. plotting probability density
    functions (PDF), or to make a second uncertainty quantification simulation
    with that newly generated Dists.

    Args:
        poly (Poly) : Polynomial of interest.
        dist (Dist) : Defines the space where the samples for the KDE is taken
                from the poly.
        sample (int) : Number of samples used in estimation to construct the
                KDE.
        **kws (optional) : Extra keywords passed to dist.sample.

    Returns:
        (ndarray) : The constructed quantity of interest (QoI) distributions,
                where `qoi_dists.shape==poly.shape`.

    Examples:
        >>> cp.seed(1000)
        >>> dist = cp.Normal(0, 1)
        >>> x = cp.variable(1)
        >>> poly = cp.Poly([x])
        >>> qoi_dist = cp.QoI_Dist(poly, dist)
        >>> values = qoi_dist[0].pdf([-0.75, 0., 0.75])
        >>> print(np.around(values, 8))
        [ 0.29143037  0.39931708  0.29536329]
    """
    shape = poly.shape
    poly = cp.poly.flatten(poly)
    dim = len(dist)

    #sample from the input dist
    samples = dist.sample(sample, **kws)

    qoi_dists = []
    for i in range(0, len(poly)):
        #sample the polynomial solution
        if dim == 1:
            dataset = poly[i](samples)
        else:
            dataset = poly[i](*samples)

        lo = dataset.min()
        up = dataset.max()

        #creates qoi_dist
        qoi_dist = cp.dist.SampleDist(dataset, lo, up)
        qoi_dists.append(qoi_dist)

    #reshape the qoi_dists to match the shape of the input poly
    qoi_dists = np.array(qoi_dists, cp.dist.Dist)
    qoi_dists = qoi_dists.reshape(shape)

    if not shape:
        qoi_dists = qoi_dists.item()

    return qoi_dists
